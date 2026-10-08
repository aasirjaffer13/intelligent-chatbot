/**
 * Thin API client for the NOVA backend.
 *
 * - Uses VITE_API_BASE_URL when set; empty string in dev relies on the
 *   Vite proxy (see vite.config.js), so the browser never hits CORS.
 * - Normalizes backend error payloads ({error:{detail}}) into thrown
 *   ApiError objects that UI code can render directly.
 * - streamChat() consumes POST /api/chat/stream (SSE over fetch) and
 *   delivers reply chunks via onDelta, resolving with the final
 *   ChatResponse contract (the `meta` event).
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function extractDetail(body, status) {
  if (body?.error?.detail) return body.error.detail
  // FastAPI 422 validation errors: detail is an array of {msg, loc, ...}
  if (Array.isArray(body?.detail)) {
    return body.detail.map((item) => item.msg).join('; ') || `Validation failed (${status})`
  }
  if (typeof body?.detail === 'string') return body.detail
  return `Request failed with status ${status}`
}

async function toApiError(response) {
  let body = null
  try {
    body = await response.json()
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(extractDetail(body, response.status), response.status)
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    })
  } catch {
    throw new ApiError('Cannot reach the NOVA backend. Is it running on port 8000?', 0)
  }

  if (!response.ok) throw await toApiError(response)
  if (response.status === 204) return null
  return response.json()
}

export function checkHealth() {
  return request('/api/health')
}

export function getStatus() {
  return request('/api/status')
}

export function sendChat(message, sessionId = null) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  })
}

/**
 * Stream a chat turn over SSE.
 *
 * The server emits `delta` events (reply chunks), then a `meta` event
 * carrying the exact ChatResponse contract, then `end`. Errors arrive as
 * an `error` event or a normal HTTP error response.
 *
 * @param {string} message
 * @param {string|null} sessionId
 * @param {{onDelta?: (text: string) => void}} handlers
 * @returns {Promise<object>} the ChatResponse contract
 */
export async function streamChat(message, sessionId = null, { onDelta } = {}) {
  let response
  try {
    response = await fetch(`${BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ message, session_id: sessionId }),
    })
  } catch {
    throw new ApiError('Cannot reach the NOVA backend. Is it running on port 8000?', 0)
  }

  if (!response.ok) throw await toApiError(response)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let meta = null
  let streamError = null

  function handleBlock(block) {
    let event = 'message'
    let data = ''
    for (const line of block.split('\n')) {
      if (line.startsWith('event: ')) event = line.slice(7)
      else if (line.startsWith('data: ')) data += line.slice(6)
    }
    if (!data) return
    let payload
    try {
      payload = JSON.parse(data)
    } catch {
      return // ignore malformed frame
    }
    if (event === 'delta') onDelta?.(payload.text)
    else if (event === 'meta') meta = payload
    else if (event === 'error') streamError = payload
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let index
    while ((index = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, index)
      buffer = buffer.slice(index + 2)
      if (block.trim()) handleBlock(block)
    }
  }
  if (buffer.trim()) handleBlock(buffer)

  if (streamError) throw new ApiError(streamError.detail ?? 'Streaming failed', 500)
  if (!meta) throw new ApiError('The stream ended without a response.', 500)
  return meta
}

export function listConversations() {
  return request('/api/conversations')
}

export function getConversationMessages(sessionId) {
  return request(`/api/conversations/${encodeURIComponent(sessionId)}/messages`)
}

export function listDocuments() {
  return request('/api/documents')
}

export async function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)
  let response
  try {
    // no Content-Type header: the browser must set the multipart boundary
    response = await fetch(`${BASE_URL}/api/documents/upload`, { method: 'POST', body: form })
  } catch {
    throw new ApiError('Cannot reach the NOVA backend. Is it running on port 8000?', 0)
  }
  if (!response.ok) throw await toApiError(response)
  return response.json()
}

export function deleteDocument(documentId) {
  return request(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' })
}
