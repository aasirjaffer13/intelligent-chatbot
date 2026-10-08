import { useEffect, useRef } from 'react'
import { Loader2, Sparkles } from 'lucide-react'
import MessageBubble from './MessageBubble.jsx'

const SUGGESTIONS = [
  'Hello, how are you?',
  'What can you do?',
  'Tell me about NOVA',
]

function EmptyState({ onSuggest }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 shadow-xl shadow-indigo-950/60">
        <Sparkles className="h-7 w-7 text-white" aria-hidden="true" />
      </div>
      <h2 className="text-lg font-semibold text-slate-100">Talk to NOVA</h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-400">
        An NLP chatbot built to learn — evolving from rule-based text processing
        toward transformers, RAG and autonomous agents.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((text) => (
          <button
            key={text}
            type="button"
            onClick={() => onSuggest(text)}
            className="rounded-full border border-slate-700 bg-slate-900 px-3.5 py-1.5 text-xs text-slate-300 transition hover:border-indigo-500 hover:bg-slate-800 hover:text-white"
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}

function HistoryLoading() {
  return (
    <div className="flex h-full items-center justify-center gap-2 text-sm text-slate-400">
      <Loader2 className="h-4 w-4 animate-spin text-indigo-400" aria-hidden="true" />
      Loading conversation…
    </div>
  )
}

export default function MessageList({
  messages,
  isSending,
  isLoadingHistory,
  onSuggest,
}) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isSending])

  if (isLoadingHistory) {
    return <HistoryLoading />
  }

  if (messages.length === 0 && !isSending) {
    return <EmptyState onSuggest={onSuggest} />
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
