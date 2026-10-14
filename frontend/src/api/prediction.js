import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000
})

export async function predictFace(file) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('explain', 'true')

  const response = await api.post('/predict', formData)

  return response.data
}