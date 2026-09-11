import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock HTMLElement.prototype.scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = vi.fn()

// Mock URL.createObjectURL and revokeObjectURL
if (!window.URL.createObjectURL) {
  window.URL.createObjectURL = vi.fn(() => 'blob:http://localhost/fake-blob')
}

if (!window.URL.revokeObjectURL) {
  window.URL.revokeObjectURL = vi.fn()
}

// Mock navigator.clipboard
if (!navigator.clipboard) {
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn(() => Promise.resolve()),
    },
    configurable: true,
  })
}
