import { Sparkles } from 'lucide-react'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="animate-fade-up flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white shadow-md shadow-indigo-950/40 sm:max-w-[70%]">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-up flex items-start gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-900">
        <Sparkles className="h-3.5 w-3.5 text-indigo-400" aria-hidden="true" />
      </div>
      <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm leading-relaxed text-slate-200 sm:max-w-[75%]">
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        {message.intent && message.intent !== 'unknown' && (
          <span className="mt-2 inline-block rounded-md bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
            intent: {message.intent} · {(message.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  )
}
