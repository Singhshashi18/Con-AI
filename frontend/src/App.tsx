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

type EntityMemoryItem = {
  id: number
  entity_name: string
  entity_type: string
  facts: string
  updated_at: string
}

type KnowledgeGraphNode = {
  id: string
  label: string
}

type KnowledgeGraphEdge = {
  id: number
  source: string
  target: string
  relation: string
  confidence: number
}

type KnowledgeGraphPayload = {
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
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
  const [insightMenuOpen, setInsightMenuOpen] = useState(false)
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
  const [historyMenuConversationId, setHistoryMenuConversationId] = useState<number | null>(null)
  const [activeRightView, setActiveRightView] = useState<'chat' | 'entity' | 'graph'>('chat')
  const [entityMemory, setEntityMemory] = useState<EntityMemoryItem[]>([])
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraphPayload>({ nodes: [], edges: [] })
  const [historySearch, setHistorySearch] = useState('')
  const messagesScrollRef = useRef<HTMLDivElement | null>(null)
  const streamAbortRef = useRef<AbortController | null>(null)

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

  const graphLayout = useMemo(() => {
    const width = 920
    const height = 500
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) * 0.36
    const count = Math.max(knowledgeGraph.nodes.length, 1)

    const positions = new Map<string, { x: number; y: number }>()
    knowledgeGraph.nodes.forEach((node, index) => {
      const angle = (index / count) * Math.PI * 2
      positions.set(node.id, {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      })
    })

    return { width, height, positions }
  }, [knowledgeGraph])

  const filteredConversations = useMemo(() => {
    const needle = historySearch.trim().toLowerCase()
    if (!needle) {
      return conversations
    }
    return conversations.filter((conversation) => {
      const inTitle = conversation.title.toLowerCase().includes(needle)
      const inLastMessage = (conversation.last_message || '').toLowerCase().includes(needle)
      return inTitle || inLastMessage
    })
  }, [conversations, historySearch])

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

  async function loadEntityMemory(conversationId: number) {
    const entities = await request<EntityMemoryItem[]>(`/conversations/${conversationId}/entity-memory`)
    setEntityMemory(entities)
  }

  async function loadKnowledgeGraph(conversationId: number) {
    const graph = await request<KnowledgeGraphPayload>(`/conversations/${conversationId}/knowledge-graph`)
    setKnowledgeGraph(graph)
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
    void Promise.all([
      loadConversationDetails(selectedConversationId),
      loadEntityMemory(selectedConversationId),
      loadKnowledgeGraph(selectedConversationId),
    ]).catch((err) => setError((err as Error).message))
  }, [selectedConversationId])

  useEffect(() => {
    const closeMenus = () => {
      setPersonaMenuOpen(false)
      setMemoryMenuOpen(false)
      setInsightMenuOpen(false)
      setHistoryMenuConversationId(null)
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
    if (sending) {
      streamAbortRef.current?.abort()
      return
    }
    if (!selectedConversationId || !inputValue.trim()) return

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
      const abortController = new AbortController()
      streamAbortRef.current = abortController

      const response = await fetch(`${API_BASE}/conversations/${conversationId}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: abortController.signal,
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

      await Promise.all([
        loadConversations(),
        loadConversationDetails(conversationId),
        loadEntityMemory(conversationId),
        loadKnowledgeGraph(conversationId),
      ])
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        // Keep partially streamed content visible when user stops generation.
        setError('')
      } else {
        setError((err as Error).message)
        await Promise.all([
          loadConversations(),
          loadConversationDetails(conversationId),
          loadEntityMemory(conversationId),
          loadKnowledgeGraph(conversationId),
        ])
      }
    } finally {
      streamAbortRef.current = null
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

  async function handleDeleteConversation(conversationId: number) {
    try {
      setError('')
      await request<{ deleted: boolean }>(`/conversations/${conversationId}`, {
        method: 'DELETE',
      })
      if (selectedConversationId === conversationId) {
        setMessages([])
        setEntityMemory([])
        setKnowledgeGraph({ nodes: [], edges: [] })
      }
      setHistoryMenuConversationId(null)
      await loadConversations()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleExportConversationPdf(conversationId: number) {
    try {
      setError('')
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/export?format=pdf`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const fileBlob = await response.blob()
      const objectUrl = window.URL.createObjectURL(fileBlob)
      const link = document.createElement('a')
      link.href = objectUrl
      link.download = `conversation_${conversationId}.pdf`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(objectUrl)
      setHistoryMenuConversationId(null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleRenameConversation(conversation: Conversation) {
    const nextTitle = window.prompt('Rename chat', conversation.title)
    if (!nextTitle || !nextTitle.trim()) return
    await patchConversation(conversation.id, { title: nextTitle.trim() })
    setHistoryMenuConversationId(null)
  }

  async function handleTogglePinConversation(conversation: Conversation) {
    await patchConversation(conversation.id, { pinned: !conversation.pinned })
    setHistoryMenuConversationId(null)
  }

  return (
    <div className={sidebarCollapsed ? 'app-shell sidebar-collapsed' : 'app-shell'}>
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
            <input
              className="sidebar-search"
              value={historySearch}
              onChange={(event) => setHistorySearch(event.target.value)}
              placeholder="Search history"
              onClick={(event) => event.stopPropagation()}
            />
            <div className="sidebar-section-label">Conversation history</div>
            <div className="history-list">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={conversation.id === selectedConversationId ? 'history-item-row active' : 'history-item-row'}
                >
                  <button
                    className={conversation.id === selectedConversationId ? 'history-item active' : 'history-item'}
                    onClick={(event) => {
                      event.stopPropagation()
                      setSelectedConversationId(conversation.id)
                      setHistoryMenuConversationId(null)
                      setActiveRightView('chat') // Always go to chat view
                    }}
                  >
                    <span className="history-title">{conversation.title}</span>
                  </button>

                  <button
                    className="history-item-menu"
                    aria-label="Open conversation actions"
                    onClick={(event) => {
                      event.stopPropagation()
                      setHistoryMenuConversationId((current) => (current === conversation.id ? null : conversation.id))
                    }}
                  >
                    •••
                  </button>

                  {historyMenuConversationId === conversation.id && (
                    <div
                      className="history-actions-menu"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <button
                        className="dropdown-item"
                        onClick={() => {
                          void handleTogglePinConversation(conversation)
                        }}
                      >
                        {conversation.pinned ? 'Unpin chat' : 'Pin chat'}
                      </button>
                      <button
                        className="dropdown-item"
                        onClick={() => {
                          void handleRenameConversation(conversation)
                        }}
                      >
                        Rename chat
                      </button>
                      <button
                        className="dropdown-item"
                        onClick={() => {
                          void handleExportConversationPdf(conversation.id)
                        }}
                      >
                        Export chat (PDF)
                      </button>
                      <button
                        className="dropdown-item danger"
                        onClick={() => {
                          void handleDeleteConversation(conversation.id)
                        }}
                      >
                        Delete chat
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </aside>

      <main className="main-stage" onClick={() => {
        setPersonaMenuOpen(false)
        setMemoryMenuOpen(false)
        setInsightMenuOpen(false)
      }}>
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="topbar-home-btn"
              onClick={(event) => {
                event.stopPropagation()
                setActiveRightView('chat')
                setPersonaMenuOpen(false)
                setMemoryMenuOpen(false)
                setInsightMenuOpen(false)
              }}
              aria-label="Go to chat view"
            >
              <span className="conversation-name">con-ai</span>
              <span className="chevron">▾</span>
            </button>

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
              <span className="menu-dot-label">Personas</span>
              <span className="menu-dot-icon">{personaMenuOpen ? '▴' : '▾'}</span>
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
                setInsightMenuOpen(false)
              }}
              aria-label="Open memory menu"
            >
              <span className="menu-dot-label">Memories</span>
              <span className="menu-dot-icon">{memoryMenuOpen ? '▴' : '▾'}</span>
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

            <button
              className="menu-dot"
              onClick={(event) => {
                event.stopPropagation()
                setInsightMenuOpen((value) => !value)
                setPersonaMenuOpen(false)
                setMemoryMenuOpen(false)
              }}
              aria-label="Open insight menu"
            >
              <span className="menu-dot-label">Visualization</span>
              <span className="menu-dot-icon">{insightMenuOpen ? '▴' : '▾'}</span>
            </button>

            {insightMenuOpen && (
              <div className="dropdown-panel dropdown-panel-far-right">
                <div className="dropdown-title">Right section</div>
                <button
                  className={activeRightView === 'entity' ? 'dropdown-item active' : 'dropdown-item'}
                  onClick={(event) => {
                    event.stopPropagation()
                    setActiveRightView('entity')
                    setInsightMenuOpen(false)
                  }}
                >
                  Entity memory
                </button>
                <button
                  className={activeRightView === 'graph' ? 'dropdown-item active' : 'dropdown-item'}
                  onClick={(event) => {
                    event.stopPropagation()
                    setActiveRightView('graph')
                    setInsightMenuOpen(false)
                  }}
                >
                  Knowledge graph
                </button>
                <button
                  className={activeRightView === 'chat' ? 'dropdown-item active' : 'dropdown-item'}
                  onClick={(event) => {
                    event.stopPropagation()
                    setActiveRightView('chat')
                    setInsightMenuOpen(false)
                  }}
                >
                  Back to chat
                </button>
              </div>
            )}
            </div>
          </div>
        </header>

        {activeRightView === 'chat' ? (
          <>
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
              <button type="submit" className="composer-send" aria-label={sending ? 'Stop generation' : 'Send message'}>
                {sending ? '■' : '➤'}
              </button>
            </form>
          </>
        ) : activeRightView === 'entity' ? (
          <section className="insight-section">
            <h3 className="insight-title">Entity Memory Dashboard</h3>
            {entityMemory.length === 0 ? (
              <div className="insight-empty">No entities found for this conversation yet.</div>
            ) : (
              <div className="entity-grid">
                {entityMemory.map((item) => (
                  <article key={item.id} className="entity-card">
                    <div className="entity-name">{item.entity_name}</div>
                    <div className="entity-type">{item.entity_type}</div>
                    <div className="entity-facts">{item.facts || 'No facts stored yet.'}</div>
                    <div className="entity-updated">Updated: {new Date(item.updated_at).toLocaleString()}</div>
                  </article>
                ))}
              </div>
            )}
          </section>
        ) : (
          <section className="insight-section">
            <h3 className="insight-title">Knowledge Graph Visualization</h3>
            {knowledgeGraph.nodes.length === 0 ? (
              <div className="insight-empty">No knowledge graph relationships found yet.</div>
            ) : (
              <div className="graph-shell">
                <svg className="graph-svg" viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}>
                  <defs>
                    <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.6)" />
                    </marker>
                  </defs>
                  {knowledgeGraph.edges.map((edge) => {
                    const src = graphLayout.positions.get(edge.source)
                    const tgt = graphLayout.positions.get(edge.target)
                    if (!src || !tgt) return null
                    const nodeRadius = 44 // bigger circle
                    const arrowMargin = 10 // extra margin outside node
                    const dx = tgt.x - src.x
                    const dy = tgt.y - src.y
                    const distance = Math.hypot(dx, dy) || 1
                    const ux = dx / distance
                    const uy = dy / distance
                    // Start arrow just outside source node, end arrow just outside target node
                    const startX = src.x + ux * (nodeRadius + arrowMargin)
                    const startY = src.y + uy * (nodeRadius + arrowMargin)
                    const endX = tgt.x - ux * (nodeRadius + arrowMargin)
                    const endY = tgt.y - uy * (nodeRadius + arrowMargin)
                    // Place label 60% along the edge, offset perpendicular for clarity
                    const labelPosX = src.x + dx * 0.6 + (-uy * 18)
                    const labelPosY = src.y + dy * 0.6 + (ux * 18)
                    return (
                      <g key={edge.id}>
                        <line x1={startX} y1={startY} x2={endX} y2={endY} className="graph-edge" markerEnd="url(#graph-arrow)" />
                        <text x={labelPosX} y={labelPosY} textAnchor="middle" className="graph-edge-label" alignmentBaseline="middle">
                          {edge.relation.replace('_', ' ')} ({edge.confidence.toFixed(2)})
                        </text>
                      </g>
                    )
                  })}
                  {knowledgeGraph.nodes.map((node) => {
                    const pos = graphLayout.positions.get(node.id)
                    if (!pos) return null
                    return (
                      <g key={node.id}>
                        <circle cx={pos.x} cy={pos.y} r="44" className="graph-node" />
                        <text x={pos.x} y={pos.y + 8} textAnchor="middle" className="graph-node-label" alignmentBaseline="middle" fontSize={18} fontWeight={700}>
                          {node.label}
                        </text>
                      </g>
                    )
                  })}
                </svg>
              </div>
            )}
          </section>
        )}

        {error && <div className="error-banner">{error}</div>}
      </main>
    </div>
  )
}

export default App
