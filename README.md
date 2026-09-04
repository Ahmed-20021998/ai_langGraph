# Dual-AI Review API (FastAPI + Vercel)

## Files
- `main.py` — FastAPI app wrapping the LangGraph pipeline (Ollama + OpenRouter dual review)
- `requirements.txt` — Python dependencies
- `vercel.json` — Vercel build/routing config

## ⚠️ Critical: Ollama cannot run on Vercel

Vercel serverless functions are stateless and short-lived — there's no
persistent process, so a local Ollama daemon cannot live there. The
`ollama` Python client normally connects to `http://127.0.0.1:11434`,
which will simply fail in a Vercel function.

**You must run Ollama somewhere that stays online**, for example:
- Your own PC/server, with a tunnel (ngrok, Cloudflare Tunnel) or a static IP
- A small VPS (DigitalOcean, Hetzner, etc.)
- A platform like Fly.io or Railway that supports long-running processes

Once Ollama is reachable over HTTPS, set the environment variable:

```
OLLAMA_HOST=https://your-ollama-host.example.com
```

`main.py` already reads this and points the Ollama client at it. If you
don't set `OLLAMA_HOST`, deployment will succeed but every `/ask`
request will fail when it tries to reach Ollama.

**Alternative:** if you don't want to manage a separate Ollama server,
replace the `ollama_generate` and `ollama_review_claude` functions with
another OpenRouter (or any hosted API) call instead of Ollama. That
removes this constraint entirely and makes the whole app a pure
HTTPS-only service, which fits Vercel's model perfectly.

## Environment variables (set in Vercel project settings)

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `OLLAMA_HOST` | Yes (unless you replace Ollama) | Public HTTPS URL of your Ollama server |

## Deploy steps

1. Install the Vercel CLI: `npm install -g vercel`
2. From this folder, run: `vercel`
3. Follow the prompts (link/create a project)
4. Add the environment variables above in the Vercel dashboard
   (Project → Settings → Environment Variables), or via CLI:
   ```
   vercel env add OPENROUTER_API_KEY
   vercel env add OLLAMA_HOST
   ```
5. Redeploy: `vercel --prod`

## Testing locally first (recommended)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-...
export OLLAMA_HOST=http://localhost:11434   # or your remote host
uvicorn main:app --reload
```

Then test:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Which is bigger, 9.11 or 9.9?"}'
```

## Timeout note

Each `/ask` request can involve up to 3 sequential LLM calls (Ollama →
OpenRouter → Ollama again on disagreement). Vercel's free (Hobby) plan
caps serverless function execution at 10 seconds, which may not be
enough for 3 chained LLM calls. Consider:
- Upgrading to a Pro plan (60s limit) or Enterprise (900s), or
- Reducing to fewer chained calls, or
- Moving to a platform without a hard timeout (Railway, Fly.io, a VPS)
  if latency becomes an issue.
