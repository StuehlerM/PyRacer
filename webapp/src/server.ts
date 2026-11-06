import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Mistral } from '@mistralai/mistralai';

const MODEL = 'mistral-small-latest';
const PORT = Number(process.env.PORT ?? 3000);
const HOST = '127.0.0.1';

const apiKey = process.env.MISTRAL_API_KEY;
if (!apiKey) {
  console.error('MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key, or export it.');
  process.exit(1);
}

const client = new Mistral({ apiKey });

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (typeof value !== 'object' || value === null) return false;
  const msg = value as Record<string, unknown>;
  return (
    (msg.role === 'system' || msg.role === 'user' || msg.role === 'assistant') &&
    typeof msg.content === 'string' &&
    msg.content.trim().length > 0
  );
}

function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((chunk) =>
        typeof chunk === 'object' && chunk !== null && 'text' in chunk
          ? String((chunk as { text: unknown }).text)
          : ''
      )
      .join('');
  }
  return '';
}

const app = express();
app.use(express.json({ limit: '256kb' }));

app.post('/api/chat', async (req, res) => {
  const messages: unknown = req.body?.messages;
  if (
    !Array.isArray(messages) ||
    messages.length === 0 ||
    messages.length > 50 ||
    !messages.every(isChatMessage)
  ) {
    return res.status(400).json({
      error: 'Expected body: { "messages": [{ "role": "system"|"user"|"assistant", "content": "..." }] }',
    });
  }

  try {
    const completion = await client.chat.complete({
      model: MODEL,
      messages: messages as ChatMessage[],
    });
    const reply = extractText(completion.choices?.[0]?.message?.content).trim();
    res.json({ reply });
  } catch (err) {
    console.error('Mistral API request failed:', err);
    res.status(502).json({ error: 'Mistral API request failed. See server logs for details.' });
  }
});

const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public');
app.use(express.static(publicDir));

app.listen(PORT, HOST, () => {
  console.log(`Chat app listening at http://${HOST}:${PORT} (model: ${MODEL})`);
});
