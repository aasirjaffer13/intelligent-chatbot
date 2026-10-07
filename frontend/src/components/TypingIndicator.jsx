export default function TypingIndicator() {
  return (
    <div className="animate-fade-up flex items-start gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-900">
        <span className="h-2 w-2 rounded-full bg-indigo-400" />
      </div>
      <div className="rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900 px-4 py-3">
        <div className="flex items-center gap-1.5" aria-label="NOVA is typing">
          <span className="dot h-2 w-2 rounded-full bg-slate-400" />
          <span className="dot h-2 w-2 rounded-full bg-slate-400" />
          <span className="dot h-2 w-2 rounded-full bg-slate-400" />
        </div>
      </div>
    </div>
  )
}
