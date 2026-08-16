# Task: make the llm-d EPP schedule WebSocket upgrades at header time

Instructions for an implementing agent. Written 2026-07-15. All file paths
are relative to the workspace root
`/usr/local/google/home/bobzetian/projects/vllmomniproject/` unless noted.
Line numbers were verified on the current checkout but treat them as
approximate — grep for the quoted anchors rather than trusting offsets.

## 1. Problem statement

The llm-d router (Envoy sidecar + EPP ext_proc server) cannot route WebSocket
upgrades. A realtime session starts as `GET /v1/realtime` with
`Upgrade: websocket`. The EPP only picks an endpoint after it has seen the
request body's `EndOfStream`, and an upgrade request keeps its stream open
forever with no body — so the EPP never responds, Envoy never gets the
`x-gateway-destination-endpoint` header it needs, and the handshake hangs
until the client times out.

This is fully diagnosed with live evidence; read these two before coding:

* `realtime/README.md` — sections "Router vs direct Service — live comparison"
  and "Why the EPP does not work for realtime".
* Captured log proof (from a debug EPP build): a plain `GET /v1/models`
  arrives with `endOfStream:true` and completes in 4 ms; the upgrade arrives
  with `endOfStream:false`, its headers are complete and intact
  (`connection: Upgrade`, `upgrade: websocket`, `sec-websocket-key`, …), and
  no further ext_proc message ever arrives for that request id.

Goal: when the EPP receives request headers that describe a WebSocket
upgrade, it must make its routing decision from the headers alone, send the
header response carrying the endpoint, and get out of the data path. After
the change, the reference client must complete a realtime session THROUGH
the router (today it only works via the direct k8s Service).

## 2. Where the code is, and one caution

* Repo: `llm-d-inference-scheduler/` (a checkout inside this workspace; Go).
* The running EPP image is a custom build,
  `us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/llm-d-router-endpoint-picker:debugext2`,
  which added two log improvements: dumping received request/response headers,
  and an `endOfStream` field on the "EPP received request" line.
  **Caution:** those debug changes are NOT present in this checkout
  (`git status` shows no modified `.go` files — they were built elsewhere).
  Implement the fix on top of the current checkout HEAD, and re-add
  equivalent debug logging as part of your change (it proved essential; keep
  the `endOfStream` field at minimum).

## 3. Root cause, with code anchors

All in `llm-d-inference-scheduler/pkg/epp/handlers/`:

1. `server.go`, `Process()` receive loop,
   `case *extProcPb.ProcessingRequest_RequestHeaders:` (~line 384; logs
   `"EPP received request"` at ~395) → calls `s.HandleRequestHeaders(...)`.
2. `request.go:38` `HandleRequestHeaders`:
   * If `req.RequestHeaders.EndOfStream` → `fallbackToRandomEndpoint(...)`
     (~line 42-49). This is why body-less plain GETs work: Envoy marks their
     headers `EndOfStream=true`.
   * Otherwise it just copies headers into `reqCtx.Request.Headers` and
     returns `nil` — no endpoint, no `reqCtx.reqHeaderResp`.
3. A WebSocket upgrade keeps the request stream open, so its headers arrive
   with `EndOfStream=false` → path 2 → `updateStateAndSendIfNeeded` has
   nothing to send (state `RequestReceived`, `reqHeaderResp == nil`,
   `server.go` ~640) → the loop blocks waiting for a `RequestBody` message
   that can never come. Deadlock, silent by design.

Also relevant, in the Envoy config the router chart ships
(`llm-d-inference-scheduler/config/charts/llm-d-router-standalone/values.yaml`
~lines 125-150): the model route already has
`upgrade_configs: [{upgrade_type: websocket}]`, and ext_proc runs with
`request_body_mode: FULL_DUPLEX_STREAMED` and
`response_body_mode: FULL_DUPLEX_STREAMED`. The second fact matters: once
the handshake succeeds, all post-upgrade WebSocket frames (both directions)
would be streamed to the EPP as body chunks unless the EPP exits the stream
first. The current body handler buffers request chunks until `EndOfStream`
(`buf.Write` in `server.go` ~line 410), which for a WebSocket would grow
unboundedly. The fix below avoids this entirely.

## 4. The fix

### 4a. The mechanism to reuse: `RequestResponseProcessingSkipped`

The codebase already has exactly the right exit path, used by parsers that
set `SkipResponseProcessing`:

* If `reqCtx.RequestState == RequestResponseProcessingSkipped`,
  `updateStateAndSendIfNeeded` (`server.go` ~line 621) sends
  `reqCtx.reqHeaderResp` (which carries the
  `x-gateway-destination-endpoint` header mutation + dynamic metadata) and
  nothing else.
* Back in `Process()` (~line 535), that state makes the handler log
  `"EPP skipped response interception, routed request"` and `return nil`,
  which **gracefully closes the ext_proc gRPC stream**. Per the comment
  there (and the Envoy ext_proc proto docs it links), Envoy then continues
  the request without consulting ext_proc for any further phase — so
  post-upgrade frames never round-trip through the EPP. No buffering leak,
  no per-frame latency tax.

### 4b. Stage 1 — proof of concept (small, do this first)

In `request.go` `HandleRequestHeaders`, after the existing `EndOfStream`
early-return and after the header-copy loop populates
`reqCtx.Request.Headers`, add:

```go
if isWebSocketUpgrade(reqCtx.Request.Headers) {
    // A websocket upgrade is a GET whose stream never ends and that has no
    // body: EndOfStream never arrives, so schedule from headers alone.
    // Processing is then skipped entirely: with FULL_DUPLEX_STREAMED body
    // modes, staying subscribed would stream every post-upgrade frame
    // through this process (and the buffered-body path would grow without
    // bound, since a websocket has no EndOfStream).
    if err := s.fallbackToRandomEndpoint(ctx, reqCtx, 0); err != nil {
        return err
    }
    reqCtx.RequestState = RequestResponseProcessingSkipped
    return nil
}
```

With the header-detection helper (note `connection` can be a comma-separated
token list, e.g. `keep-alive, Upgrade`, and values are case-insensitive;
keys in `reqCtx.Request.Headers` are already lowercased):

```go
func isWebSocketUpgrade(headers map[string]string) bool {
    if !strings.EqualFold(strings.TrimSpace(headers["upgrade"]), "websocket") {
        return false
    }
    for _, tok := range strings.Split(headers["connection"], ",") {
        if strings.EqualFold(strings.TrimSpace(tok), "upgrade") {
            return true
        }
    }
    return false
}
```

That alone should make the handshake complete end-to-end: random endpoint
pick, header response sent, ext_proc stream closed, Envoy upgrades and pins
the connection via ORIGINAL_DST.

Add a log line for the decision (mirroring the existing style), e.g.
`"EPP scheduled websocket upgrade at header time"` with the target endpoint
— the live test below greps for it.

### 4c. Stage 2 — real scheduling instead of the random pick

Replace `fallbackToRandomEndpoint` with the same machinery the body path
uses (`server.go` `RequestBody` case, ~lines 415-460), minus the body:

1. Resolve the parser: `s.getOrResolveParser(ctx, reqCtx)`. On this stack the
   profile is `payload-agnostic.yaml`, whose parser is the passthrough parser
   (`pkg/epp/framework/plugins/requesthandling/parsers/passthrough/`) — it
   converts any body, including empty, to a RawPayload without inspecting it.
2. `parseResult, err := parser.ParseRequest(ctx, nil, reqCtx.Request.Headers)`
   (empty body). Verify the passthrough parser tolerates a nil/empty body —
   its unit test `passthrough_test.go` suggests it does, but confirm.
3. `reqCtx, err = s.director.HandleRequest(ctx, reqCtx, parseResult.Body)` —
   this runs admission, the scheduling profile (scorers incl.
   `active-request-scorer`), and `PreRequest` plugins (in-flight counter
   increment), and sets `reqCtx.TargetEndpoint`.
4. `reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(ctx, reqCtx)`,
   set `RequestResponseProcessingSkipped`, return.
5. On any error from steps 1-3, fall back to `fallbackToRandomEndpoint` (an
   upgrade should degrade to random placement, not 400) — log when this
   happens.

**Known accounting limitation — document it, don't solve it here.** The
in-flight counter (`inflightload` producer) is decremented when the producer's
`ResponseBody` hook sees `EndOfStream`
(`pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/producer.go`,
`ResponseBody`, `PluginState.Delete` on `resp.EndOfStream`). Because this fix
closes the ext_proc stream at header time, the EPP never observes the end of
the websocket session, so the in-flight entry is only released by the
PluginState TTL/eviction. Consequence: least-active-sessions balancing sees
sessions as "active" for the TTL duration rather than their true lifetime.
That is acceptable for this iteration; note it in code comments and the PR
description. (The alternative — keeping the ext_proc stream open and passing
every frame through just to observe session end — costs a per-frame proxy
round-trip and needs new passthrough body handling; a cleaner future answer
is deriving session counts from model-server metrics.)

### 4d. Tests

* Unit tests in `pkg/epp/handlers/` for `isWebSocketUpgrade` (mixed case,
  token lists, missing headers) and for the upgrade branch (headers with
  `EndOfStream=false` + upgrade headers → response contains an endpoint and
  state is `RequestResponseProcessingSkipped`; without upgrade headers →
  unchanged behavior). Follow the existing test style in that package.
* `go build ./...` and the package tests
  (`go test ./pkg/epp/handlers/... ./pkg/epp/framework/plugins/requesthandling/...`)
  must pass. Note: some pre-existing tests in unrelated packages may be
  flaky/broken — only gate on packages you touched plus a full `go build`.
* Do NOT break the `EndOfStream=true` fallback path (plain GETs) or the
  normal body path (POST chat/completions) — both have existing coverage.

## 5. Build and push the image

From `llm-d-inference-scheduler/`:

```bash
# Auth for Artifact Registry (once per machine):
gcloud auth configure-docker us-central1-docker.pkg.dev

# The Makefile builds $(IMAGE_REGISTRY)/llm-d-router-endpoint-picker:$(EPP_TAG)
IMAGE_REGISTRY=us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference \
  EPP_TAG=upgradefix1 \
  make image-build-epp

docker push us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/llm-d-router-endpoint-picker:upgradefix1
```

(If the make target misbehaves, `Dockerfile.epp` at the repo root is a
standard multi-stage build — `docker build -f Dockerfile.epp -t <ref> .`
works too. There may also be an `image-push-epp`-style target; check
`make help`.)

## 6. Deploy to the live stack

The stack runs in namespace `llm-d-realtime`; its values file already pins a
custom EPP image. Change only the tag:

* Edit `realtime/router/realtime.values.yaml` → `router.epp.image.tag:
  upgradefix1` (leave registry/repository/pullPolicy as they are).

```bash
cd /usr/local/google/home/bobzetian/projects/vllmomniproject
source llm-d/guides/env.sh
helm upgrade llm-d-realtime $ROUTER_STANDALONE_CHART \
  -f llm-d/guides/recipes/router/base.values.yaml \
  -f realtime/router/realtime.values.yaml \
  -n llm-d-realtime --version $ROUTER_CHART_VERSION
kubectl rollout status -n llm-d-realtime deploy/llm-d-realtime-epp --timeout=180s
```

Do not touch the model-server Deployment — it is already Running (Qwen3-Omni
takes ~20-40 min to reload; there is no reason to restart it).

## 7. Live acceptance test

```bash
# Port-forward the ROUTER (not the direct Service):
kubectl port-forward -n llm-d-realtime svc/llm-d-realtime-epp 8082:80 &

# 1. Plain HTTP regression check — must still return the model:
curl -sm 10 http://localhost:8082/v1/models | jq -r '.data[].id'

# 2. The acceptance test — this HANGS today and must COMPLETE after the fix:
python3 vllm-omni/examples/online_serving/qwen3_omni/openai_realtime_client.py \
  --url ws://localhost:8082/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-wav realtime/input_16k_mono.wav \
  --output-wav /tmp/router_fixed_output.wav
```

Pass criteria:

* The client completes a full session (prints streamed transcription text
  and `Saved realtime audio to: ...`; the output WAV is nonempty — expect
  tens of seconds of 24 kHz PCM16 for the 440 Hz test tone). Allow up to
  ~4 min: replies are long. `timeout 300` is a reasonable wrapper.
* EPP logs (`kubectl logs -n llm-d-realtime deploy/llm-d-realtime-epp -c epp`)
  show, for the upgrade request id: the received headers, your
  "scheduled websocket upgrade at header time" line with an endpoint, and
  the "EPP skipped response interception, routed request" line. No body
  chunks for that request id afterward.
* The pre-fix failure signature is GONE: no request id whose last line is
  the received-headers dump.
* Regression: request 1 (`/v1/models`) still returns 200 through the router,
  and the direct-Service path still works:
  `kubectl port-forward -n llm-d-realtime svc/optimized-baseline-realtime-nvidia-gpu-vllm-direct 8083:80`
  then the same client against `ws://localhost:8083/v1/realtime`.

## 8. When done

* Update `realtime/README.md`: the "This WebSocket step currently HANGS"
  warning, the live-comparison table, and the "Why the EPP does not work"
  section need a dated addendum saying the custom build fixes it (do not
  delete the historical evidence — it documents stock v0.9.0 behavior).
* Update `docs/realtime-voice-interaction.md` §4 and
  `docs/routing-opportunities.md` (the 2026-07-06 realtime update block)
  the same way.
* Commit the scheduler changes on a branch in `llm-d-inference-scheduler/`
  with a message framing it as the upstream contribution target: "schedule
  connection-upgrade requests at header time".
