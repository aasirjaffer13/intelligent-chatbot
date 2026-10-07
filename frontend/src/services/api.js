/**
 * Thin API client for the NOVA backend.
 *
 * - Uses VITE_API_BASE_URL when set; empty string in dev relies on the
 *   Vite proxy (see vite.config.js), so the browser never hits CORS.
 * - Normalizes backend error payloads ({error:{detail}}) into thrown
 *   ApiError objects that UI code can render directly.
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

  if (!response.ok) {
    let body = null
    try {
      body = await response.json()
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(extractDetail(body, response.status), response.status)
  }

  return response.json()
}

export function checkHealth() {
  return request('/api/health')
}

export function sendChat(message, sessionId = null) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  })
}
