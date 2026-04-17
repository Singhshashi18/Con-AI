import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

type MemoryStrategy = 'buffer' | 'summary' | 'entity' | 'knowledge_graph' | 'summary_entity'

type Persona = {
  id: number
  name: string
  avatar: string
  system_prompt: string
  personality: string
  default_memory_strategy: MemoryStrategy
  recommended_memory_strategy: MemoryStrategy
  temperature: number
  domain_focus: string
}

type Conversation = {
  id: number
  title: string
  memory_strategy: MemoryStrategy
  pinned: boolean
  persona_id: number | null
  persona_name: string | null
  persona_avatar: string
  message_count: number
  last_message: string
  updated_at: string
}

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

const API_BASE = '/api'
const PERSONA_TYPES = ['General Assistant', 'Code Helper', 'Creative Writer', 'Business Analyst', 'Study Buddy']
const MEMORY_TYPES: MemoryStrategy[] = ['buffer', 'summary', 'entity', 'knowledge_graph', 'summary_entity']
const STREAM_RENDER_DELAY_MS = 42

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json() as Promise<T>
}

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [personaMenuOpen, setPersonaMenuOpen] = useState(false)
  const [memoryMenuOpen, setMemoryMenuOpen] = useState(false)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [personas, setPersonas] = useState<Persona[]>([])
  const [selectedConversationId, setSelectedConversationId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [sending, setSending] = useState(false)
  const [activePersonaId, setActivePersonaId] = useState<number | ''>('')
  const [activeMemory, setActiveMemory] = useState<MemoryStrategy>('buffer')
  const [error, setError] = useState('')
  const [streamingAssistantId, setStreamingAssistantId] = useState<number | null>(null)
  const [awaitingFirstToken, setAwaitingFirstToken] = useState(false)
  const messagesScrollRef = useRef<HTMLDivElement | null>(null)

  const selectedPersona = personas.find((persona) => persona.id === activePersonaId) || null
  const visibleMessages = useMemo(() => {
    return messages.filter((message, index, list) => {
      if (message.role !== 'assistant' || index === 0) {
        return true
      }

      const previous = list[index - 1]
      if (!previous || previous.role !== 'assistant') {
        return true
      }

      const currentContent = message.content.trim()
      const previousContent = previous.content.trim()
      if (!currentContent || !previousContent) {
        return true
      }

      return currentContent !== previousContent
    })
  }, [messages])

  async function loadPersonas() {
    const data = await request<Persona[]>('/personas')
    setPersonas(data)
    if (data.length > 0 && activePersonaId === '') {
      setActivePersonaId(data[0].id)
    }
  }

  async function loadConversations() {
    const data = await request<Conversation[]>('/conversations')
    setConversations(data)
    if (data.length === 0) {
      setSelectedConversationId(null)
      return
    }

    if (!selectedConversationId || !data.some((conversation) => conversation.id === selectedConversationId)) {
      setSelectedConversationId(data[0].id)
    }
  }

  async function loadConversationDetails(conversationId: number) {
    const chat = await request<ChatMessage[]>(`/conversations/${conversationId}/messages`)
    setMessages(chat)
  }

  useEffect(() => {
    void (async () => {
      try {
        await Promise.all([loadPersonas(), loadConversations()])
      } catch (err) {
        setError((err as Error).message)
      }
    })()
  }, [])

  useEffect(() => {
    if (!selectedConversationId) return
    void loadConversationDetails(selectedConversationId).catch((err) => setError((err as Error).message))
  }, [selectedConversationId])

  useEffect(() => {
    const closeMenus = () => {
      setPersonaMenuOpen(false)
      setMemoryMenuOpen(false)
    }

    window.addEventListener('click', closeMenus)
    return () => window.removeEventListener('click', closeMenus)
  }, [])

  useEffect(() => {
    if (activePersonaId === '' && personas.length > 0) {
      setActivePersonaId(personas[0].id)
    }
  }, [personas, activePersonaId])

  useEffect(() => {
    if (!messagesScrollRef.current) return
    messagesScrollRef.current.scrollTop = messagesScrollRef.current.scrollHeight
  }, [messages])

  async function createConversation() {
    try {
      setError('')
      const persona = personas.find((item) => item.id === activePersonaId) || null
      const created = await request<Conversation>('/conversations', {
        method: 'POST',
        body: JSON.stringify({
          title: 'New Chat',
          persona_id: persona?.id || null,
          memory_strategy: activeMemory,
        }),
      })
      await loadConversations()
      setSelectedConversationId(created.id)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleSend(event: FormEvent) {
    event.preventDefault()
    if (!selectedConversationId || !inputValue.trim() || sending) return

    const conversationId = selectedConversationId
    const text = inputValue.trim()
    const tempUserId = Date.now()
    const tempAssistantId = tempUserId + 1

    try {
      setError('')
      setSending(true)
      setInputValue('')
      setMessages((previous) => [
        ...previous,
        { id: tempUserId, role: 'user', content: text, created_at: new Date().toISOString() },
        { id: tempAssistantId, role: 'assistant', content: '', created_at: new Date().toISOString() },
      ])
      setStreamingAssistantId(tempAssistantId)
      setAwaitingFirstToken(true)

      let streamFailed = false
      let streamCompleted = false
      let receivedToken = false

      const response = await fetch(`${API_BASE}/conversations/${conversationId}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })

      if (!response.ok || !response.body) {
        streamFailed = true
      } else {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        const processSseBlock = async (block: string) => {
          const dataLines = block
            .split('\n')
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trim())

          if (dataLines.length === 0) return

          const rawPayload = dataLines.join('')
          if (!rawPayload) return

          let payload: { type?: string; token?: string; error?: string } = {}
          try {
            payload = JSON.parse(rawPayload) as { type?: string; token?: string; error?: string }
          } catch {
            return
          }

          if (payload.type === 'token' && payload.token) {
            receivedToken = true
            setAwaitingFirstToken(false)
            setMessages((previous) =>
              previous.map((msg) =>
                msg.id === tempAssistantId ? { ...msg, content: `${msg.content}${payload.token}` } : msg,
              ),
            )
            await wait(STREAM_RENDER_DELAY_MS)
            return
          }

          if (payload.type === 'done') {
            streamCompleted = true
            return
          }

          if (payload.type === 'error') {
            streamFailed = true
          }
        }

        while (true) {
          const { value, done } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const blocks = buffer.split('\n\n')
          buffer = blocks.pop() || ''

          for (const block of blocks) {
            await processSseBlock(block)
            if (streamFailed) break
          }

          if (streamFailed) break
        }

        // Process any remaining partial SSE block after stream closes.
        if (buffer.trim()) {
          await processSseBlock(buffer)
        }
      }

      // Fallback only when stream clearly failed before any usable token.
      if (streamFailed || (!streamCompleted && !receivedToken)) {
        setAwaitingFirstToken(false)
        setMessages((previous) => previous.filter((msg) => msg.id !== tempAssistantId))
      }

      await Promise.all([loadConversations(), loadConversationDetails(conversationId)])
    } catch (err) {
      setError((err as Error).message)
      await Promise.all([loadConversations(), loadConversationDetails(conversationId)])
    } finally {
      setStreamingAssistantId(null)
      setAwaitingFirstToken(false)
      setSending(false)
    }
  }

  async function patchConversation(conversationId: number, patch: Record<string, unknown>) {
    try {
      setError('')
      await request(`/conversations/${conversationId}`, {
        method: 'PATCH',
        body: JSON.stringify(patch),
      })
      await loadConversations()
      await loadConversationDetails(conversationId)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-top">
          <button
            className="collapse-btn"
            onClick={(event) => {
              event.stopPropagation()
              setSidebarCollapsed((value) => !value)
            }}
            aria-label="Collapse sidebar"
          >
            {sidebarCollapsed ? '›' : '‹'}
          </button>
          {!sidebarCollapsed && <span className="sidebar-title">con-ai</span>}
        </div>

        <button className="new-chat-btn" onClick={createConversation}>
          <span>＋</span>
          {!sidebarCollapsed && <span>New chat</span>}
        </button>

        {!sidebarCollapsed && (
          <>
            <div className="sidebar-section-label">Conversation history</div>
            <div className="history-list">
              {conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  className={conversation.id === selectedConversationId ? 'history-item active' : 'history-item'}
                  onClick={(event) => {
                    event.stopPropagation()
                    setSelectedConversationId(conversation.id)
                  }}
                >
                  <span className="history-title">{conversation.title}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </aside>

      <main className="main-stage" onClick={() => {
        setPersonaMenuOpen(false)
        setMemoryMenuOpen(false)
      }}>
        <header className="topbar">
          <div className="topbar-left">
            <span className="conversation-name">con-ai</span>
            <span className="chevron">▾</span>

            <div className="topbar-right">
            <button
              className="menu-dot"
              onClick={(event) => {
                event.stopPropagation()
                setPersonaMenuOpen((value) => !value)
                setMemoryMenuOpen(false)
              }}
              aria-label="Open persona menu"
            >
              •••
            </button>

            {personaMenuOpen && (
              <div className="dropdown-panel">
                <div className="dropdown-title">Persona types</div>
                {personas.length === 0 ? (
                  PERSONA_TYPES.map((persona) => (
                    <button key={persona} className="dropdown-item" onClick={(event) => event.stopPropagation()}>
                      {persona}
                    </button>
                  ))
                ) : (
                  personas.map((persona) => (
                    <button
                      key={persona.id}
                      className={persona.id === activePersonaId ? 'dropdown-item active' : 'dropdown-item'}
                      onClick={(event) => {
                        event.stopPropagation()
                        setActivePersonaId(persona.id)
                        if (selectedConversationId) {
                          void patchConversation(selectedConversationId, { persona_id: persona.id })
                        }
                        setPersonaMenuOpen(false)
                      }}
                    >
                      {persona.name}
                    </button>
                  ))
                )}
              </div>
            )}

            <button
              className="menu-dot"
              onClick={(event) => {
                event.stopPropagation()
                setMemoryMenuOpen((value) => !value)
                setPersonaMenuOpen(false)
              }}
              aria-label="Open memory menu"
            >
              •••
            </button>

            {memoryMenuOpen && (
              <div className="dropdown-panel dropdown-panel-right">
                <div className="dropdown-title">Memory types</div>
                {MEMORY_TYPES.map((memory) => (
                  <button
                    key={memory}
                    className={memory === activeMemory ? 'dropdown-item active' : 'dropdown-item'}
                    onClick={(event) => {
                      event.stopPropagation()
                      setActiveMemory(memory)
                      if (selectedConversationId) {
                        void patchConversation(selectedConversationId, { memory_strategy: memory })
                      }
                      setMemoryMenuOpen(false)
                    }}
                  >
                    {memory.replace('_', ' ')}
                  </button>
                ))}
              </div>
            )}
            </div>
          </div>
        </header>

        <section className="chat-area">
          <div className="messages-scroll" ref={messagesScrollRef}>
            {visibleMessages.map((message) => (
              <div key={message.id} className={`chat-row ${message.role}`}>
                <div className={`chat-bubble ${message.role}`}>
                  {message.role === 'assistant' && message.id === streamingAssistantId && awaitingFirstToken && !message.content.trim() ? (
                    <div className="thinking-wrap" aria-label="Assistant is thinking">
                      <span className="thinking-logo" aria-hidden="true" />
                      <span className="thinking-dots" aria-hidden="true">
                        <span />
                        <span />
                        <span />
                      </span>
                    </div>
                  ) : (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="status-row">
            <div className="status-pill">Persona: {selectedPersona?.name || 'General Assistant'}</div>
            <div className="status-pill">Memory: {activeMemory.replace('_', ' ')}</div>
          </div>
        </section>

        <form className="composer-shell" onSubmit={handleSend}>
          <button type="button" className="composer-icon">＋</button>
          <input
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder="Ask anything"
          />
          <button type="submit" className="composer-send" disabled={sending}>
            ➤
          </button>
        </form>

        {error && <div className="error-banner">{error}</div>}
      </main>
    </div>
  )
}

export default App
