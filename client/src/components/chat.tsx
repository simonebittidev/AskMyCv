'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowUpIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';

const suggestedActions = [
  {
    label: "Experiences",
    text: "Which companies has Simone worked for?",
  },
  {
    label: "Technologies",
    text: "What technologies does Simone have experience with?",
  },
  {
    label: "Personal projects",
    text: "What personal projects has Simone worked on?",
  },
  {
    label: "Contact",
    text: "How can I contact Simone?",
  }
];

function getRandomItems<T>(arr: T[], n: number): T[] {
  const shuffled = arr.slice().sort(() => 0.5 - Math.random());
  return shuffled.slice(0, n);
}

type ChatMessage = {
  role: 'human' | 'ai';
  content: string;
};

const mdComponents: Components = {
  p:          (props) => <p className="my-2 leading-relaxed" {...props} />,
  strong:     (props) => <strong className="font-semibold text-ink" {...props} />,
  em:         (props) => <em className="italic" {...props} />,
  a:          (props) => <a className="text-accent underline underline-offset-2 hover:text-ink" target="_blank" rel="noopener noreferrer" {...props} />,
  ul:         (props) => <ul className="my-2 ml-5 list-disc space-y-1" {...props} />,
  ol:         (props) => <ol className="my-2 ml-5 list-decimal space-y-1" {...props} />,
  li:         (props) => <li className="leading-relaxed" {...props} />,
  code:       (props) => <code className="rounded bg-warm/60 px-1 py-0.5 text-[0.9em] font-mono" {...props} />,
  pre:        (props) => <pre className="my-3 overflow-x-auto rounded-lg bg-ink text-cream p-3 text-xs font-mono" {...props} />,
  h1:         (props) => <h1 className="mt-3 mb-2 text-xl font-semibold" {...props} />,
  h2:         (props) => <h2 className="mt-3 mb-2 text-lg font-semibold" {...props} />,
  h3:         (props) => <h3 className="mt-2 mb-1 text-base font-semibold" {...props} />,
  hr:         (props) => <hr className="my-4 border-warm" {...props} />,
  blockquote: (props) => <blockquote className="my-2 border-l-2 border-accent-soft pl-3 text-muted" {...props} />,
};

const ChatContent = () => {
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isInputFocused, setInputFocused] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [aiError, setAiError] = useState(false);

  const handleSuggestedClick = (text: string) => {
      setChatMessages(prev => [
        ...prev,
        { role: 'human', content: text },
        { role: 'ai', content: '' }
      ]);
      startStream(text);
  };

  const startStream = (text: string) => {
    setAiError(false);

    eventSourceRef.current?.close();

    const encodedHistory = encodeURIComponent(JSON.stringify(chatMessages.slice(-3)));
    const eventSource = new EventSource(`/stream?text=${encodeURIComponent(text)}&history=${encodedHistory}`);
    eventSourceRef.current = eventSource;

    let currentMessage = "";

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        eventSource.close();
        if (eventSourceRef.current === eventSource) {
          eventSourceRef.current = null;
        }
        return;
      }

      const data = JSON.parse(event.data);
      currentMessage += data.content;

      setChatMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg?.role === 'ai') {
          updated[updated.length - 1] = {
            ...lastMsg,
            content: currentMessage,
          };
          return updated;
        }
        return [...updated, { role: "ai", content: data.content }];
      });
    };

    eventSource.onerror = () => {
      console.error("Errore nella ricezione SSE");
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
    };
  };

  const handleSend = () => {
    if (!chatInput.trim()) return;

    const text = chatInput.trim();
    setChatMessages((prev) => [...prev, { role: 'human', content: text }, { role: 'ai', content: '' }]);
    setChatInput('');
    startStream(text);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, []);

  const MOBILE_WIDTH = 768;
  const isMobile = typeof window !== "undefined" && window.innerWidth < MOBILE_WIDTH;

  const [actionsToShow] = useState<{
    label: string;
    text: string;
  }[]>(() => {
    return isMobile
      ? getRandomItems(suggestedActions, 2)
      : suggestedActions;
  });

  return (
    <div className="flex flex-col h-screen w-full bg-cream text-ink">
      <main className="flex-1 overflow-y-auto" ref={chatBoxRef}>
        <div
          className="mx-auto max-w-2xl px-4 pt-6"
          style={{ paddingBottom: isInputFocused && isMobile ? "260px" : "160px" }}
        >
          {!chatMessages.length && (
            <div className="mt-8 mb-4 text-center">
              <h2 className="text-2xl font-semibold text-ink">Hello there!</h2>
              <p className="mt-3 text-muted leading-relaxed">
                I&apos;m here to help you explore Simone&apos;s professional profile.<br />
                Interested in his experience, skills, or what sets him apart?<br />
                Just ask a question — I&apos;ll guide you through his CV.
              </p>
            </div>
          )}

          {chatMessages.map((msg, i) => (
            msg.role === 'human' ? (
              <div key={i} className="mb-6 flex justify-end">
                <div className="rounded-[18px] bg-ink text-cream px-4 py-2.5 max-w-[80%] text-[15px] whitespace-pre-line leading-relaxed">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div key={i} className="mb-6">
                {msg.content === '[ERROR]' ? (
                  <div className="text-[15px] text-accent leading-relaxed">
                    An error occurred with the service. Please try again later.
                  </div>
                ) : !msg.content ? (
                  <div aria-label="Thinking" className="space-y-2 py-1">
                    <div className="shimmer-bar h-3 w-3/4" />
                    <div className="shimmer-bar h-3 w-1/2" />
                  </div>
                ) : (
                  <div className="text-[15px] text-ink">
                    <ReactMarkdown components={mdComponents}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )
          ))}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <div className="fixed bottom-0 inset-x-0 z-30 pb-3 pt-6 bg-gradient-to-t from-cream via-cream/90 to-transparent">
        <div className="mx-auto max-w-2xl px-4">
          {!chatMessages.length && (
            <div data-testid="suggested-actions" className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full mb-3">
              {actionsToShow.map((action, i) => (
                <button
                  key={action.label}
                  onClick={() => handleSuggestedClick(action.text)}
                  className={`rounded-2xl border border-warm bg-surface/70 hover:bg-warm transition-colors text-left px-4 py-3 ${!isMobile && i > 1 ? 'hidden sm:block' : ''}`}
                >
                  <div className="text-sm font-medium text-ink lowercase">{action.label}</div>
                  <div className="text-xs text-muted mt-0.5">{action.text}</div>
                </button>
              ))}
            </div>
          )}

          <div tabIndex={0}>
            <div className="flex items-end gap-2 rounded-full glass border border-warm/60 px-3 py-2 shadow-sm">
              <textarea
                ref={inputRef}
                value={chatInput}
                rows={1}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask a follow-up…"
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                className="flex-1 bg-transparent resize-none outline-none px-2 py-2 text-[15px] text-ink placeholder:text-muted-2 max-h-32"
                onFocus={() => {
                  setInputFocused(true);
                  inputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                }}
                onBlur={() => {
                  setInputFocused(false);
                  setTimeout(() => {
                    inputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }, 100);
                }}
              />
              <button
                onClick={handleSend}
                aria-label="Send"
                className="w-9 h-9 flex-shrink-0 rounded-full bg-ink text-cream flex items-center justify-center hover:opacity-90 transition-opacity"
              >
                <ArrowUpIcon className="w-4 h-4" />
              </button>
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-2">
              This is a personal study project; answers may be inaccurate.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatContent;
