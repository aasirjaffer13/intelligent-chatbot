import { useCallback, useEffect, useRef, useState } from 'react'
import {
  checkHealth,
  getConversationMessages,
  listConversations,
  streamChat,
} from '../services/api.js'

let messageSeq = 0

function makeMessage(role, content, extra = {}) {
  messageSeq += 1
  return { id: `m-${Date.now()}-${messageSeq}`, role, content, ...extra }
}

/**
 * Chat state machine: messages, streaming assistant turns, conversation
 * list, in-flight flag, error banner. All backend I/O lives here so
 * components stay presentational.
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState(null)
  const [conversations, setConversations] = useState([])
  const [conversationsLoading, setConversationsLoading] = useState(true)
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const sessionIdRef = useRef(null)

  const send = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed) return

    setError(null)
    const placeholder = makeMessage('assistant', '', { streaming: true })
    setMessages((prev) => [...prev, makeMessage('user', trimmed), placeholder])
    setIsSending(true)

    const patchAssistant = (patch) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === placeholder.id ? { ...m, ...patch } : m)),
      )

    try {
      const meta = await streamChat(trimmed, sessionIdRef.current, {
        onDelta: (chunk) =>
          setMessages((prev) =>
            prev.map((m) =>
              m.id === placeholder.id ? { ...m, content: m.content + chunk } : m,
            ),
          ),
      })
      sessionIdRef.current = meta.session_id ?? sessionIdRef.current
      setActiveSessionId(sessionIdRef.current)
      patchAssistant({
        content: meta.response,
        intent: meta.intent,
        confidence: meta.confidence,
        sources: meta.sources,
        streaming: false,
      })
    } catch (err) {
      setError(err.message ?? 'Something went wrong. Please try again.')
      // drop the placeholder if nothing streamed; otherwise keep partial text
      setMessages((prev) =>
        prev.flatMap((m) => {
          if (m.id !== placeholder.id) return [m]
          return m.content === '' ? [] : [{ ...m, streaming: false }]
        }),
      )
    } finally {
      setIsSending(false)
    }
  }, [])

  const clearChat = useCallback(() => {
    setMessages([])
    setError(null)
    sessionIdRef.current = null // new conversation, not a wipe of the old one
    setActiveSessionId(null)
  }, [])

  const refreshConversations = useCallback(async () => {
    try {
      const data = await listConversations()
      setConversations(data.conversations)
    } catch {
      /* sidebar degrades quietly — chat still works */
    } finally {
      setConversationsLoading(false) // skeleton only on first load
    }
  }, [])

  const loadConversation = useCallback(async (sessionId) => {
    setIsLoadingHistory(true)
    setError(null)
    try {
      const data = await getConversationMessages(sessionId)
      setMessages(
        data.messages.map((m, index) => ({
          id: `h-${sessionId}-${index}`,
          role: m.role,
          content: m.content,
          intent: m.intent,
          confidence: m.confidence,
        })),
      )
      sessionIdRef.current = sessionId
      setActiveSessionId(sessionId)
    } catch (err) {
      setError(err.message ?? 'Could not load that conversation.')
    } finally {
      setIsLoadingHistory(false)
    }
  }, [])

  const dismissError = useCallback(() => setError(null), [])

  return {
    messages,
    isSending,
    error,
    send,
    clearChat,
    dismissError,
    conversations,
    conversationsLoading,
    refreshConversations,
    activeSessionId,
    loadConversation,
    isLoadingHistory,
  }
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
