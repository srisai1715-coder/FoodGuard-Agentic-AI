import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_ATTACHMENTS = 5
const MAX_FILE_SIZE = 10 * 1024 * 1024

function createAttachmentId(file) {
  return `${file.name}-${file.size}-${file.lastModified}-${file.type}`
}

const ImageUpload = forwardRef(function ImageUpload({ attachments, onAttachmentsChange }, ref) {
  const inputRef = useRef(null)
  const attachmentsRef = useRef(attachments)
  const [error, setError] = useState('')

  useEffect(() => {
    attachmentsRef.current = attachments
  }, [attachments])

  useEffect(() => {
    return () => {
      attachmentsRef.current.forEach((attachment) => {
        URL.revokeObjectURL(attachment.previewUrl)
      })
    }
  }, [])

  function openFilePicker() {
    inputRef.current?.click()
  }

  useImperativeHandle(ref, () => ({
    openPicker() {
      openFilePicker()
    },
  }))

  function handleFileSelection(event) {
    const selectedFiles = Array.from(event.target.files ?? [])
    event.target.value = ''

    if (selectedFiles.length === 0) {
      return
    }

    const existingIds = new Set(attachments.map((attachment) => attachment.id))
    const newAttachments = []
    const errors = []

    for (const file of selectedFiles) {
      const id = createAttachmentId(file)

      if (!ACCEPTED_TYPES.includes(file.type)) {
        errors.push(
          `${file.name} is not a supported image type. Use JPG, JPEG, PNG, or WEBP.`
        )
        continue
      }

      if (file.size > MAX_FILE_SIZE) {
        errors.push(
          `${file.name} is too large. Each image must be 10 MB or smaller.`
        )
        continue
      }

      if (
        existingIds.has(id) ||
        newAttachments.some((attachment) => attachment.id === id)
      ) {
        errors.push(`${file.name} is already attached.`)
        continue
      }

      newAttachments.push({
        id,
        file,
        previewUrl: URL.createObjectURL(file),
        name: file.name,
        type: file.type,
        size: file.size,
      })
    }

    const availableSlots = MAX_ATTACHMENTS - attachments.length

    if (newAttachments.length > availableSlots) {
      newAttachments.slice(availableSlots).forEach((attachment) => {
        URL.revokeObjectURL(attachment.previewUrl)
      })

      newAttachments.length = availableSlots
      errors.push('Maximum 5 images can be attached at a time.')
    }

    onAttachmentsChange([...attachments, ...newAttachments])
    setError(errors.join(' '))
  }

  function removeAttachment(id) {
    const attachment = attachments.find((item) => item.id === id)

    if (attachment) {
      URL.revokeObjectURL(attachment.previewUrl)
    }

    onAttachmentsChange(attachments.filter((item) => item.id !== id))
    setError('')
  }

  return (
    <div className="attachment-area">
      <div className="attachment-toolbar">
        <div className="attachment-previews" aria-live="polite">
          {attachments.map((attachment) => (
            <div
              className="attachment-preview"
              key={attachment.id}
              style={{
                position: 'relative',
                width: '64px',
                height: '64px',
                flex: '0 0 64px',
                overflow: 'hidden',
                borderRadius: '8px',
                border: '1px solid #cbd8ce',
                background: '#f3f0e8',
              }}
            >
              <img
                src={attachment.previewUrl}
                alt={attachment.name}
                style={{
                  display: 'block',
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />

              <button
                type="button"
                className="remove-attachment"
                aria-label={`Remove ${attachment.name}`}
                title={`Remove ${attachment.name}`}
                onClick={() => removeAttachment(attachment.id)}
                style={{
                  position: 'absolute',
                  top: '2px',
                  right: '2px',
                  width: '20px',
                  height: '20px',
                  padding: 0,
                  display: 'grid',
                  placeItems: 'center',
                  border: 0,
                  borderRadius: '50%',
                  background: 'rgba(33, 51, 59, 0.85)',
                  color: '#ffffff',
                  fontSize: '14px',
                  lineHeight: 1,
                  cursor: 'pointer',
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <button
          type="button"
          className="attachment-trigger"
          aria-label="Attach food product images"
          title="Attach food product images"
          onClick={openFilePicker}
        >
          +
        </button>
      </div>

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        multiple
        onChange={handleFileSelection}
      />

      {error && (
        <p className="attachment-error" role="alert">
          {error}
        </p>
      )}

      <p className="attachment-hint">Up to 5 images, 10 MB each</p>
    </div>
  )
})

export default ImageUpload