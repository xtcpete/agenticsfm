# Shared VLM service: Qwen3.8-27B on vgldgx01

A vision-language model with thinking and tool calling, served by vLLM behind an
OpenAI-compatible API. Anyone on the server, or on the lab network, can use it: there is no key.

| | |
|---|---|
| Endpoint (on vgldgx01) | `http://localhost:8000/v1` |
| Endpoint (other machines) | `http://172.16.34.150:8000/v1` (or `http://vgldgx01:8000/v1`) |
| Model name | `Qwen/Qwen3.8-27B` |
| API key | none; pass any string (`EMPTY`) if your client insists on one |
| Context | 131,072 tokens per request |
| Images | up to 24 per request (base64 data URLs or http URLs the server can fetch) |
| GPUs | 0, 1, 2, 3 (tensor parallel 4, bf16); shared with the agenticsfm experiments |
| Server | vLLM 0.19.0, run by gochen in a tmux session |

## Is it up?

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool | head
curl -s http://localhost:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)'
```

The first prints the model entry; the second shows the current load. If `curl` cannot connect,
the server is down: message gochen (see "Restarting" at the end).

## curl

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B",
  "messages": [{"role": "user", "content": "Explain bundle adjustment in three sentences."}],
  "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## Python (openai package)

`pip install openai` in any environment, then:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

r = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Explain bundle adjustment in three sentences."}],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

Any OpenAI-compatible client works the same way (LangChain's `ChatOpenAI`, LiteLLM, Continue,
Open WebUI ...): set the base URL, the model name and a dummy key.

## Thinking mode

The model can reason before it answers. Thinking is on by default; the reasoning is returned
separately from the answer in `message.reasoning` (or `reasoning_content`, depending on the client
version) and the answer in `message.content`. Thinking tokens count against `max_tokens`, so give it
room (the harness uses 16,384).

```python
r = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Is 2^31 - 1 prime? Answer yes or no with one reason."}],
    max_tokens=8192,
    temperature=1.0, top_p=0.95,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}},
)
msg = r.choices[0].message
print("thinking:", getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None))
print("answer:", msg.content)
```

Sampling settings from the model card: thinking on, temperature 1.0, top_p 0.95, top_k 20;
thinking off, temperature 0.7, top_p 0.8, top_k 20. Avoid greedy decoding (temperature 0) with
thinking on; it tends to loop. `reasoning_effort` is `low`, `medium` or `high`.

## Images

```python
import base64, pathlib

def image_url(path):
    data = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}

r = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": [
        image_url("a.jpg"), image_url("b.jpg"),
        {"type": "text", "text": "Do these two photos show the same building? Which side of it?"},
    ]}],
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

Downscale images before sending (about 1000 px on the long side is plenty); a large image costs
thousands of prompt tokens and slows everyone's requests.

## Tool calling

The server parses tool calls, so the standard OpenAI `tools` / `tool_choice` API works:

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
r = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is the weather in Zurich?"}],
    tools=tools, max_tokens=2048,
)
print(r.choices[0].message.tool_calls)
```

Append the assistant message and a `{"role": "tool", "tool_call_id": ..., "content": ...}` message
with the result, and call again, as with the OpenAI API.

## Streaming, logprobs, batching

`stream=True` works. `logprobs=True, top_logprobs=k` works up to k = 20. vLLM batches
concurrent requests, so send several in parallel (threads or asyncio) instead of one after the
other; the GPUs are shared, so keep sustained jobs to a few concurrent requests and prefer short
contexts. The embeddings and completions (non-chat) endpoints are not enabled.

## Being a good neighbour

- The server is shared with running agent experiments (the agenticsfm harness). A burst of
  long-context requests slows them; big batch jobs are welcome at night.
- Do not restart or kill the server. If it is down or misbehaving, message gochen.
- There is no authentication and no logging of your prompts beyond vLLM's request log on the host.

## Restarting (gochen only)

The server runs in tmux session `0` on vgldgx01 from
`/raid/home/gochen/cvpr_2027/agenticsfm/scripts/serve_vlm.sh` (`CUDA_VISIBLE_DEVICES=0,1,2,3`,
`PORT=8000`, `HOST=0.0.0.0`). Loading takes about 5 minutes.

```bash
tmux attach -t 0            # ctrl-b d to detach
bash /raid/home/gochen/cvpr_2027/agenticsfm/scripts/serve_vlm.sh
```
