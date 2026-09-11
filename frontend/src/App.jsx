import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ImageUpload from './components/ImageUpload'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [attachments, setAttachments] = useState([])
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRestoring, setIsRestoring] = useState(false)
  const [error, setError] = useState('')
  const [processingStatus, setProcessingStatus] = useState('')

  const [activeBarcode, setActiveBarcode] = useState(null)
  const [activeComparisonBarcodes, setActiveComparisonBarcodes] = useState([])
  const [conversationId, setConversationId] = useState(null)

  const [copiedIndex, setCopiedIndex] = useState(null)
  const [lightboxImage, setLightboxImage] = useState(null)

  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const imageUploadRef = useRef(null)

  const [ocrContext, setOcrContext] = useState({
    ocr_combined_text: null,
    ocr_nutrition_text: null,
    ingredient_text: null,
    allergen_statement: null,
    cross_contact_statement: null,
    nutrition_data: null,
  })

  // --------------------------------------------------
  // Auto-scroll to bottom of conversation
  // --------------------------------------------------
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading, isRestoring])

  // --------------------------------------------------
  // Auto-adjust textarea height
  // --------------------------------------------------
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [message])

  // --------------------------------------------------
  // Revoke object URLs on component unmount
  // --------------------------------------------------
  useEffect(() => {
    return () => {
      messages.forEach((chatMessage) => {
        if (chatMessage.images) {
          chatMessage.images.forEach((image) => {
            if (image.url) {
              URL.revokeObjectURL(image.url)
            }
          })
        }
      })
    }
  }, [messages])

  // --------------------------------------------------
  // Lightbox keyboard navigation (Escape to close)
  // --------------------------------------------------
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setLightboxImage(null)
      }
    }
    if (lightboxImage) {
      window.addEventListener('keydown', handleKeyDown)
    }
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [lightboxImage])

  // --------------------------------------------------
  // Restore conversation from backend (GET /conversations/{cid})
  // --------------------------------------------------
  const restoreConversation = useCallback(async (cid) => {
    if (!cid) return
    setIsRestoring(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE_URL}/conversations/${encodeURIComponent(cid)}`)

      if (!response.ok) {
        if (response.status === 404 || response.status === 400) {
          throw new Error('Conversation not found or expired.')
        }
        throw new Error('Could not restore conversation from server.')
      }

      const data = await response.json()

      setConversationId(data.conversation_id)
      setActiveBarcode(data.active_barcode || null)
      setActiveComparisonBarcodes(data.active_comparison_barcodes || [])

      setOcrContext({
        ocr_combined_text: data.ocr_combined_text || null,
        ocr_nutrition_text: data.ocr_nutrition_text || null,
        ingredient_text: null,
        allergen_statement: null,
        cross_contact_statement: null,
        nutrition_data: null,
      })

      if (Array.isArray(data.messages)) {
        setMessages(
          data.messages.map((m) => ({
            role: m.role,
            content: m.content,
            timestamp: m.created_at
              ? new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : null,
          }))
        )
      }
    } catch (err) {
      setError(err.message || 'Failed to restore conversation.')
    } finally {
      setIsRestoring(false)
    }
  }, [])

  // --------------------------------------------------
  // Initial URL check for ?cid=...
  // --------------------------------------------------
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const cid = params.get('cid')
    if (cid && cid !== conversationId) {
      restoreConversation(cid)
    }
  }, [conversationId, restoreConversation])

  // --------------------------------------------------
  // Handle browser Back / Forward navigation (popstate)
  // --------------------------------------------------
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search)
      const cid = params.get('cid')
      if (cid && cid !== conversationId) {
        restoreConversation(cid)
      } else if (!cid && conversationId) {
        startNewChat(false)
      }
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [conversationId, restoreConversation])

  // --------------------------------------------------
  // Reset all ephemeral conversation state for New Chat
  // --------------------------------------------------
  function startNewChat(updateUrl = true) {
    setMessages([])
    setAttachments([])
    setOcrContext({
      ocr_combined_text: null,
      ocr_nutrition_text: null,
      ingredient_text: null,
      allergen_statement: null,
      cross_contact_statement: null,
      nutrition_data: null,
    })
    setActiveBarcode(null)
    setActiveComparisonBarcodes([])
    setConversationId(null)
    setError('')
    setProcessingStatus('')

    if (updateUrl && window.location.search) {
      window.history.replaceState({}, '', window.location.pathname)
    }
  }

  // --------------------------------------------------
  // Update temporary conversation context from response
  // --------------------------------------------------
  function updateConversationContext(data) {
    setOcrContext({
      ocr_combined_text: data.ocr_combined_text ?? null,
      ocr_nutrition_text: data.ocr_nutrition_text ?? null,
      ingredient_text: data.ingredient_text ?? null,
      allergen_statement: data.allergen_statement ?? null,
      cross_contact_statement: data.cross_contact_statement ?? null,
      nutrition_data: data.nutrition_data ?? null,
    })

    if (data.barcode) {
      setActiveBarcode(data.barcode)
    }

    if (data.comparison_barcodes && data.comparison_barcodes.length >= 2) {
      setActiveComparisonBarcodes(data.comparison_barcodes)
    }

    if (data.conversation_id) {
      setConversationId(data.conversation_id)

      // Synchronize ?cid=<conversation_id> into the browser URL
      const params = new URLSearchParams(window.location.search)
      if (params.get('cid') !== data.conversation_id) {
        const newUrl = `${window.location.pathname}?cid=${encodeURIComponent(data.conversation_id)}`
        window.history.replaceState({ cid: data.conversation_id }, '', newUrl)
      }
    }
  }

  // --------------------------------------------------
  // Send message
  // --------------------------------------------------
  async function sendMessage(event) {
    if (event) {
      event.preventDefault()
    }

    const content = message.trim()
    const hasNewImages = attachments.length > 0

    if ((!content && !hasNewImages) || isLoading || isRestoring) {
      return
    }

    setError('')

    const uploadedImages = hasNewImages
      ? attachments.map((attachment) => ({
          name: attachment.name,
          url: URL.createObjectURL(attachment.file),
        }))
      : []

    const userMessageContent = content || (hasNewImages ? 'Analyze this product label image.' : '')
    const currentTimeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

    const conversationMessages = [
      ...messages.map((chatMessage) => ({
        role: chatMessage.role,
        content: chatMessage.content,
      })),
      {
        role: 'user',
        content: userMessageContent,
      },
    ]

    setMessages((currentMessages) => [
      ...currentMessages,
      {
        role: 'user',
        content: userMessageContent,
        images: uploadedImages,
        timestamp: currentTimeStr,
      },
    ])

    setMessage('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    setProcessingStatus(hasNewImages ? 'Processing image(s)...' : '')
    setIsLoading(true)

    try {
      const request = {
        message: userMessageContent,
        messages: conversationMessages,
        barcode: activeBarcode,
        barcodes: activeComparisonBarcodes,
        conversation_id: conversationId,
        ...ocrContext,
      }

      const options = {
        method: 'POST',
      }

      if (hasNewImages) {
        const formData = new FormData()

        formData.append('message', userMessageContent)
        formData.append('messages', JSON.stringify(conversationMessages))

        if (activeBarcode) {
          formData.append('barcode', activeBarcode)
        }

        if (activeComparisonBarcodes && activeComparisonBarcodes.length > 0) {
          formData.append('barcodes', JSON.stringify(activeComparisonBarcodes))
        }

        if (conversationId) {
          formData.append('conversation_id', conversationId)
        }

        if (ocrContext.ocr_combined_text) {
          formData.append('ocr_combined_text', ocrContext.ocr_combined_text)
        }

        if (ocrContext.ocr_nutrition_text) {
          formData.append('ocr_nutrition_text', ocrContext.ocr_nutrition_text)
        }

        if (ocrContext.ingredient_text) {
          formData.append('ingredient_text', ocrContext.ingredient_text)
        }

        if (ocrContext.allergen_statement) {
          formData.append('allergen_statement', ocrContext.allergen_statement)
        }

        if (ocrContext.cross_contact_statement) {
          formData.append('cross_contact_statement', ocrContext.cross_contact_statement)
        }

        if (ocrContext.nutrition_data) {
          formData.append('nutrition_data', JSON.stringify(ocrContext.nutrition_data))
        }

        attachments.forEach((attachment) => {
          formData.append('images', attachment.file, attachment.name)
        })

        options.body = formData
      } else {
        options.headers = {
          'Content-Type': 'application/json',
        }
        options.body = JSON.stringify(request)
      }

      const response = await fetch(`${API_BASE_URL}/chat`, options)

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null)
        throw new Error(errorBody?.detail || 'The assistant could not respond right now.')
      }

      const data = await response.json()

      updateConversationContext(data)

      if (hasNewImages) {
        setAttachments([])
      }

      if (data.ocr_status) {
        setProcessingStatus(
          data.ocr_warnings?.length
            ? `OCR completed with ${data.ocr_warnings.length} warning(s).`
            : `OCR completed: ${data.ocr_status.toLowerCase()}.`
        )
      } else {
        setProcessingStatus('')
      }

      const resTimeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: 'assistant',
          content: data.response,
          timestamp: resTimeStr,
        },
      ])
    } catch (requestError) {
      setError(requestError.message || 'The assistant could not respond right now.')
      setProcessingStatus('')

      uploadedImages.forEach((image) => {
        if (image.url) {
          URL.revokeObjectURL(image.url)
        }
      })
    } finally {
      setIsLoading(false)
    }
  }

  // --------------------------------------------------
  // Textarea keyboard listener (Enter to send, Shift+Enter for newline)
  // --------------------------------------------------
  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // --------------------------------------------------
  // Copy assistant response to clipboard
  // --------------------------------------------------
  function copyResponseText(text, index) {
    if (navigator.clipboard) {
      navigator.clipboard
        .writeText(text)
        .then(() => {
          setCopiedIndex(index)
          setTimeout(() => setCopiedIndex(null), 2000)
        })
        .catch(() => {})
    }
  }

  // --------------------------------------------------
  // Starter Prompt Click Handler
  // --------------------------------------------------
  function handleQuickPrompt(promptText) {
    setMessage(promptText)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  return (
    <main className="app-shell">
      {/* --------------------------------------------------
          Header / Introduction
          -------------------------------------------------- */}
      <header className="intro" aria-labelledby="page-title">
        <span className="eyebrow">FoodGuard AI</span>
        <h1 id="page-title">Your conversational food analysis assistant</h1>
        <p className="intro-copy">
          Analyze ingredients, compare food products, and make informed safety and nutrition choices.
        </p>
      </header>

      {/* --------------------------------------------------
          Main Chat Panel
          -------------------------------------------------- */}
      <section className="chat-panel" aria-label="Food analysis chat">
        {/* Panel Header with Status & New Chat Button */}
        <div className="chat-status-bar">
          <div className="status-indicator">
            <span className="status-dot" aria-hidden="true" />
            <span className="status-label">
              {isRestoring
                ? 'Restoring conversation...'
                : isLoading
                ? 'Analyzing...'
                : 'Ready when you are'}
            </span>
          </div>

          {/* Active Context Badges */}
          <div className="context-badges" aria-label="Active session context">
            {activeComparisonBarcodes.length >= 2 && (
              <span className="context-badge comparison" title={`Comparing barcodes: ${activeComparisonBarcodes.join(', ')}`}>
                ⚖️ Comparing 2 Products
              </span>
            )}
            {activeBarcode && activeComparisonBarcodes.length < 2 && (
              <span className="context-badge barcode" title={`Active Barcode: ${activeBarcode}`}>
                📦 Product {activeBarcode}
              </span>
            )}
            {ocrContext.ocr_combined_text && !activeBarcode && activeComparisonBarcodes.length < 2 && (
              <span className="context-badge ocr" title="Product Label OCR Context Active">
                📷 OCR Context Active
              </span>
            )}
          </div>

          {(messages.length > 0 || activeBarcode || activeComparisonBarcodes.length > 0 || ocrContext.ocr_combined_text || conversationId) && (
            <button
              type="button"
              className="new-chat-button"
              onClick={() => startNewChat(true)}
              aria-label="Start a new conversation"
            >
              + New Chat
            </button>
          )}
        </div>

        {/* Messages or Empty State */}
        <div className={`message-container ${messages.length ? 'has-messages' : ''}`}>
          {isRestoring ? (
            <div className="restoring-state">
              <div className="spinner" aria-hidden="true" />
              <p>Restoring conversation from FoodGuard database...</p>
            </div>
          ) : messages.length === 0 ? (
            <div className="empty-state">
              <div className="placeholder-mark" aria-hidden="true">🛡️</div>
              <h2>How can FoodGuard AI help you today?</h2>
              <p className="empty-copy">
                Try scanning a barcode, uploading a product label image, comparing two items, or asking a food safety question.
              </p>

              {/* Starter Quick Prompts */}
              <div className="starter-prompts" aria-label="Suggested starter prompts">
                <button
                  type="button"
                  className="prompt-chip"
                  onClick={() => handleQuickPrompt('Can you analyze barcode 3017620422003?')}
                >
                  <span className="chip-icon">🔍</span>
                  <span>Analyze Nutella (3017620422003)</span>
                </button>

                <button
                  type="button"
                  className="prompt-chip"
                  onClick={() => handleQuickPrompt('Compare barcode 3017620422003 with barcode 0051500255162')}
                >
                  <span className="chip-icon">⚖️</span>
                  <span>Compare Nutella vs Jif Peanut Butter</span>
                </button>

                <button
                  type="button"
                  className="prompt-chip"
                  onClick={() => handleQuickPrompt('What are the health and safety risks of high fructose corn syrup?')}
                >
                  <span className="chip-icon">🛡️</span>
                  <span>Ask Food Safety & Additive Question</span>
                </button>

                <button
                  type="button"
                  className="prompt-chip"
                  onClick={() => {
                    if (imageUploadRef.current) {
                      imageUploadRef.current.openPicker()
                    }
                  }}
                >
                  <span className="chip-icon">📷</span>
                  <span>Upload Product Label Image</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="chat-messages" aria-live="polite">
              {messages.map((chatMessage, index) => (
                <div className={`chat-message ${chatMessage.role}`} key={`${chatMessage.role}-${index}`}>
                  <div className="message-header">
                    <span className="role-label">
                      {chatMessage.role === 'user' ? 'You' : 'FoodGuard AI'}
                    </span>
                    {chatMessage.timestamp && (
                      <span className="timestamp">{chatMessage.timestamp}</span>
                    )}
                  </div>

                  {/* Uploaded Images */}
                  {chatMessage.images?.length > 0 && (
                    <div className="chat-message-images">
                      {chatMessage.images.map((image, imageIndex) => (
                        <button
                          key={`${image.name}-${imageIndex}`}
                          type="button"
                          className="image-thumbnail-btn"
                          onClick={() => setLightboxImage(image)}
                          title={`Click to view ${image.name}`}
                          aria-label={`View full image ${imageIndex + 1}`}
                        >
                          <img
                            src={image.url}
                            alt={`Uploaded food product label ${imageIndex + 1}`}
                          />
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Message Content */}
                  <div className="message-body">
                    {chatMessage.role === 'assistant' ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {chatMessage.content}
                      </ReactMarkdown>
                    ) : (
                      <div className="user-text">{chatMessage.content}</div>
                    )}
                  </div>

                  {/* Copy Button for Assistant Messages */}
                  {chatMessage.role === 'assistant' && (
                    <button
                      type="button"
                      className="copy-btn"
                      onClick={() => copyResponseText(chatMessage.content, index)}
                      aria-label="Copy assistant response"
                      title="Copy response to clipboard"
                    >
                      {copiedIndex === index ? 'Copied! ✓' : 'Copy'}
                    </button>
                  )}
                </div>
              ))}

              {/* Animated Typing Indicator */}
              {isLoading && (
                <div className="chat-message assistant loading">
                  <div className="message-header">
                    <span className="role-label">FoodGuard AI</span>
                  </div>
                  <div className="typing-indicator" aria-label="FoodGuard AI is thinking">
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                  <span className="sr-only">FoodGuard AI is analyzing...</span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Form */}
        <form className="message-form" onSubmit={sendMessage}>
          <label htmlFor="message-input" className="visually-hidden">
            Message FoodGuard AI
          </label>

          <ImageUpload
            ref={imageUploadRef}
            attachments={attachments}
            onAttachmentsChange={setAttachments}
          />

          <div className="input-row">
            <textarea
              id="message-input"
              ref={textareaRef}
              rows={1}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about food safety, scan a barcode, or compare items..."
              disabled={isLoading || isRestoring}
              aria-label="Message FoodGuard AI"
            />

            <button
              type="submit"
              disabled={isLoading || isRestoring || (!message.trim() && attachments.length === 0)}
              aria-label="Send message"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>

          {error && (
            <p className="chat-error" role="alert">
              ⚠️ {error}
            </p>
          )}

          {processingStatus && (
            <p className="ocr-status" role="status">
              ℹ️ {processingStatus}
            </p>
          )}
        </form>
      </section>

      {/* --------------------------------------------------
          Persistent Safety Disclaimer Footer
          -------------------------------------------------- */}
      <footer className="safety-disclaimer" role="contentinfo">
        <p>
          🛡️ <strong>FoodGuard AI Safety Notice:</strong> Provides educational food, nutrition, and product insights. It is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
      </footer>

      {/* --------------------------------------------------
          Image Lightbox Modal
          -------------------------------------------------- */}
      {lightboxImage && (
        <div
          className="lightbox-overlay"
          onClick={() => setLightboxImage(null)}
          role="dialog"
          aria-modal="true"
          aria-label="Full size image preview"
        >
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="lightbox-close"
              onClick={() => setLightboxImage(null)}
              aria-label="Close image preview"
            >
              ×
            </button>
            <img src={lightboxImage.url} alt={lightboxImage.name} />
            <span className="lightbox-caption">{lightboxImage.name}</span>
          </div>
        </div>
      )}
    </main>
  )
}

export default App