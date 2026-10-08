import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CodeBlock from './CodeBlock.jsx'

/**
 * Renders assistant markdown: paragraphs, lists, links, tables, quotes
 * and fenced code (with copy buttons via CodeBlock). Styled by hand so
 * every color flows through Tailwind vars — it follows dark/light for free.
 */
const markdownComponents = {
  p: ({ children }) => <p className="my-1.5 leading-relaxed first:mt-0 last:mb-0">{children}</p>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="font-medium text-indigo-400 underline decoration-indigo-400/40 underline-offset-2 hover:decoration-indigo-400"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => <ul className="my-1.5 list-disc space-y-1 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal space-y-1 pl-5">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  h1: ({ children }) => <h3 className="mt-3 mb-1.5 text-base font-semibold text-slate-100 first:mt-0">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-3 mb-1.5 text-sm font-semibold text-slate-100 first:mt-0">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-2.5 mb-1 text-sm font-semibold text-slate-100 first:mt-0">{children}</h4>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-indigo-500 pl-3 text-slate-300 italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-slate-700" />,
  table: ({ children }) => (
    <div className="scroll-slim my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-slate-700 bg-slate-800 px-2 py-1 text-left font-semibold text-slate-200">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border border-slate-700 px-2 py-1 text-slate-300">{children}</td>,
  pre: ({ children }) => <>{children}</>,
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className ?? '')
    const text = String(children).replace(/\n$/, '')
    if (match || text.includes('\n')) {
      return <CodeBlock language={match?.[1]} code={text} />
    }
    return (
      <code
        className="rounded bg-slate-800 px-1 py-0.5 font-mono text-[0.85em] text-indigo-300"
        {...props}
      >
        {children}
      </code>
    )
  },
}

export default function MarkdownContent({ content }) {
  return (
    <div className="break-words text-sm [&>*:first-child]:mt-0 [&>*:first-child]:mb-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  )
}
