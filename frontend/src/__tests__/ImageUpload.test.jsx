import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ImageUpload from '../components/ImageUpload'

describe('ImageUpload Component', () => {
  it('renders trigger button and attachment hints', () => {
    render(<ImageUpload attachments={[]} onAttachmentsChange={vi.fn()} />)

    expect(screen.getByRole('button', { name: /attach food product images/i })).toBeInTheDocument()
    expect(screen.getByText(/up to 5 images, 10 mb each/i)).toBeInTheDocument()
  })

  it('handles valid file selection', async () => {
    const onAttachmentsChange = vi.fn()

    render(<ImageUpload attachments={[]} onAttachmentsChange={onAttachmentsChange} />)

    const fileInput = document.querySelector('input[type="file"]')
    const validFile = new File(['dummy content'], 'label.jpg', { type: 'image/jpeg' })

    fireEvent.change(fileInput, { target: { files: [validFile] } })

    expect(onAttachmentsChange).toHaveBeenCalled()
    const newAttachments = onAttachmentsChange.mock.calls[0][0]
    expect(newAttachments.length).toBe(1)
    expect(newAttachments[0].name).toBe('label.jpg')
  })

  it('rejects unsupported file formats', () => {
    const onAttachmentsChange = vi.fn()

    render(<ImageUpload attachments={[]} onAttachmentsChange={onAttachmentsChange} />)

    const fileInput = document.querySelector('input[type="file"]')
    const invalidFile = new File(['pdf text'], 'document.pdf', { type: 'application/pdf' })

    fireEvent.change(fileInput, { target: { files: [invalidFile] } })

    expect(screen.getByText(/document.pdf is not a supported image type/i)).toBeInTheDocument()
    expect(onAttachmentsChange).toHaveBeenCalledWith([])
  })

  it('rejects files larger than 10MB', () => {
    const onAttachmentsChange = vi.fn()

    render(<ImageUpload attachments={[]} onAttachmentsChange={onAttachmentsChange} />)

    const fileInput = document.querySelector('input[type="file"]')
    const hugeFile = new File([new ArrayBuffer(11 * 1024 * 1024)], 'huge.png', { type: 'image/png' })

    fireEvent.change(fileInput, { target: { files: [hugeFile] } })

    expect(screen.getByText(/huge.png is too large/i)).toBeInTheDocument()
    expect(onAttachmentsChange).toHaveBeenCalledWith([])
  })

  it('allows removing attachments', async () => {
    const onAttachmentsChange = vi.fn()
    const user = userEvent.setup()

    const mockAttachments = [
      {
        id: 'file1-100-1234-image/jpeg',
        name: 'file1.jpg',
        previewUrl: 'blob:http://localhost/file1',
        file: new File([''], 'file1.jpg', { type: 'image/jpeg' }),
      },
    ]

    render(<ImageUpload attachments={mockAttachments} onAttachmentsChange={onAttachmentsChange} />)

    const removeBtn = screen.getByRole('button', { name: /remove file1.jpg/i })
    await user.click(removeBtn)

    expect(onAttachmentsChange).toHaveBeenCalledWith([])
  })
})
