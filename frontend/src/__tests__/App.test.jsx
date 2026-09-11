import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

describe('App Main Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    window.history.replaceState({}, '', '/')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders app shell, title, empty state, and starter prompts', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /your conversational food analysis assistant/i })).toBeInTheDocument()
    expect(screen.getByText(/how can foodguard ai help you today\?/i)).toBeInTheDocument()
    expect(screen.getByText(/analyze nutella \(3017620422003\)/i)).toBeInTheDocument()
    expect(screen.getByText(/compare nutella vs jif peanut butter/i)).toBeInTheDocument()
  })

  it('submits a text message and renders assistant Markdown response', async () => {
    const mockChatResponse = {
      response: '### Nutella Analysis\n- High in sugar\n- Contains **hazelnuts**',
      conversation_id: 'conv-123',
      barcode: '3017620422003',
    }

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (url.toString().includes('/chat')) {
        return {
          ok: true,
          json: async () => mockChatResponse,
        }
      }
      return { ok: false, status: 404 }
    })

    const user = userEvent.setup()
    render(<App />)

    const textarea = screen.getByLabelText(/message foodguard ai/i)
    await user.type(textarea, 'Can you analyze barcode 3017620422003?')

    const sendBtn = screen.getByRole('button', { name: /send message/i })
    await user.click(sendBtn)

    expect(screen.getByText('Can you analyze barcode 3017620422003?')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /nutella analysis/i })).toBeInTheDocument()
      expect(screen.getByText(/high in sugar/i)).toBeInTheDocument()
      expect(screen.getByText(/hazelnuts/i)).toBeInTheDocument()
    })

    expect(screen.getByText(/product 3017620422003/i)).toBeInTheDocument()
    expect(window.location.search).toContain('cid=conv-123')
  })

  it('populates textarea when starter quick prompt chip is clicked', async () => {
    const user = userEvent.setup()
    render(<App />)

    const promptChip = screen.getByText(/ask food safety & additive question/i)
    await user.click(promptChip)

    const textarea = screen.getByLabelText(/message foodguard ai/i)
    expect(textarea.value).toContain('health and safety risks of high fructose corn syrup')
  })

  it('restores conversation history when loaded with ?cid=<conversation_id>', async () => {
    window.history.replaceState({}, '', '/?cid=test-conv-999')

    const mockRestoredData = {
      conversation_id: 'test-conv-999',
      active_barcode: '3017620422003',
      active_comparison_barcodes: [],
      ocr_combined_text: null,
      messages: [
        { role: 'user', content: 'What is in Nutella?', created_at: '2026-09-09T10:00:00Z' },
        { role: 'assistant', content: 'Nutella contains sugar and palm oil.', created_at: '2026-09-09T10:00:05Z' },
      ],
    }

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (url.toString().includes('/conversations/test-conv-999')) {
        return {
          ok: true,
          json: async () => mockRestoredData,
        }
      }
      return { ok: false, status: 404 }
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('What is in Nutella?')).toBeInTheDocument()
      expect(screen.getByText('Nutella contains sugar and palm oil.')).toBeInTheDocument()
      expect(screen.getByText(/product 3017620422003/i)).toBeInTheDocument()
    })
  })

  it('handles invalid or 404 ?cid= by displaying friendly error message', async () => {
    window.history.replaceState({}, '', '/?cid=invalid-uuid-123')

    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Conversation not found.' }),
      }
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/conversation not found or expired/i)).toBeInTheDocument()
    })
  })

  it('clears React state and URL when + New Chat is clicked', async () => {
    window.history.replaceState({}, '', '/?cid=active-cid-555')

    const mockRestoredData = {
      conversation_id: 'active-cid-555',
      active_barcode: '3017620422003',
      active_comparison_barcodes: [],
      messages: [
        { role: 'user', content: 'Hello', created_at: '2026-09-09T10:00:00Z' },
        { role: 'assistant', content: 'Hi there!', created_at: '2026-09-09T10:00:05Z' },
      ],
    }

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (url.toString().includes('/conversations/active-cid-555')) {
        return {
          ok: true,
          json: async () => mockRestoredData,
        }
      }
      return { ok: false, status: 404 }
    })

    const user = userEvent.setup()
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument()
    })

    const newChatBtn = screen.getByRole('button', { name: /start a new conversation/i })
    await user.click(newChatBtn)

    expect(screen.queryByText('Hello')).toBeNull()
    expect(screen.getByText(/how can foodguard ai help you today\?/i)).toBeInTheDocument()
    expect(window.location.search).toBe('')
  })
})
