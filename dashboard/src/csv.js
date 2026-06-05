import Papa from 'papaparse'

// Parse a CSV File into an array of row objects, optionally dropping the first
// `skipRows` lines first (the raw acoustic recordings have ~5 metadata rows
// before the real header).
export function parseCsvFile(file, { skipRows = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      let text = reader.result
      if (skipRows > 0) {
        text = String(text).split(/\r?\n/).slice(skipRows).join('\n')
      }
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (out) => resolve(out.data),
        error: reject,
      })
    }
    reader.onerror = () => reject(new Error('Could not read the file.'))
    reader.readAsText(file)
  })
}
