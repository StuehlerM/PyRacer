# PyRacer Web App — Mistral Chat

A minimal web app that chats with the [Mistral API](https://docs.mistral.ai) using the
official TypeScript SDK (`@mistralai/mistralai`). The browser only talks to the local
server's `/api/chat` endpoint; **`MISTRAL_API_KEY` never leaves the server** (it is read
from the environment and is git-ignored).

- Model: `mistral-small-latest`
- Stack: Node.js + TypeScript + Express, vanilla HTML/JS frontend
- Server binds to `127.0.0.1` (localhost only)

## Install & run locally

Requires Node.js 22+.

```bash
cd webapp
npm install

cp .env.example .env        # then edit .env and paste your MISTRAL_API_KEY

npm run dev                 # open http://localhost:3000
```

Alternatively, export the key instead of using a `.env` file:

```bash
export MISTRAL_API_KEY=your-mistral-api-key-here
npm run dev
```

## Production-style run

```bash
npm run build               # compiles src/ -> dist/ with tsc
npm start                   # serves dist/server.js on http://localhost:3000
```

`PORT` can be overridden via the environment (default `3000`).

## How it works

1. `public/index.html` keeps the conversation in the browser and POSTs it to `/api/chat`.
2. `src/server.ts` validates the messages, calls `client.chat.complete()` with
   `mistral-small-latest`, and returns `{ "reply": "..." }`.
3. The API key is only ever read from `process.env.MISTRAL_API_KEY` on the server.

## Files

```
webapp/
  src/server.ts     Express server + Mistral SDK call (/api/chat)
  public/index.html Chat UI
  .env.example      Template for the API key (copy to .env, never commit .env)
  package.json      Dependencies and scripts
  tsconfig.json     TypeScript config
```
