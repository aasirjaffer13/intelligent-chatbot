import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, sendChat } from '../services/api.js'

let messageSeq = 0

function makeMessage(role, content, extra = {}) {
  messageSeq += 1
  return { id: `m-${Date.now()}-${messageSeq}`, role, content, ...extra }
}

/**
 * Chat state machine: messages, in-flight flag, error banner.
 * All backend I/O lives here so components stay presentational.
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState(null)
  const sessionIdRef = useRef(null)

  const send = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed) return

    setError(null)
    setMessages((prev) => [...prev, makeMessage('user', trimmed)])
    setIsSending(true)

    try {
      const data = await sendChat(trimmed, sessionIdRef.current)
      sessionIdRef.current = data.session_id ?? sessionIdRef.current
      setMessages((prev) => [
        ...prev,
        makeMessage('assistant', data.response, {
          intent: data.intent,
          confidence: data.confidence,
        }),
      ])
    } catch (err) {
      setError(err.message ?? 'Something went wrong. Please try again.')
    } finally {
      setIsSending(false)
    }
  }, [])

  const clearChat = useCallback(() => {
    setMessages([])
    setError(null)
    sessionIdRef.current = null // new conversation, not a wipe of the old one
  }, [])

  const dismissError = useCallback(() => setError(null), [])

  return { messages, isSending, error, send, clearChat, dismissError }
}

/** Polls GET /api/health so the header can show a live status badge. */
export function useHealth(intervalMs = 30_000) {
  const [status, setStatus] = useState('checking') // 'checking' | 'online' | 'offline'

  useEffect(() => {
    let cancelled = false

    async function check() {
      try {
        const data = await checkHealth()
        if (!cancelled && data.status === 'healthy') setStatus('online')
        else if (!cancelled) setStatus('offline')
      } catch {
        if (!cancelled) setStatus('offline')
      }
    }

    check()
    const timer = setInterval(check, intervalMs)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [intervalMs])

  return status
}
