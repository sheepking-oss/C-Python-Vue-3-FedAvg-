import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const CLIENT_CONFIGS = [
  { id: 'client_1', port: 5001, name: '客户端 1' },
  { id: 'client_2', port: 5002, name: '客户端 2' },
  { id: 'client_3', port: 5003, name: '客户端 3' }
]

const SERVER_URL = 'http://localhost:8080'

export const useClientsStore = defineStore('clients', () => {
  const clients = ref([])
  const lossHistory = ref({})
  const isLoading = ref(false)
  const serverStatus = ref(null)
  const pollInterval = ref(null)

  const init = async () => {
    clients.value = CLIENT_CONFIGS.map(config => ({
      ...config,
      status: null,
      isOnline: false,
      lastUpdate: null
    }))

    for (const client of clients.value) {
      lossHistory.value[client.id] = []
    }

    await checkAllStatus()
  }

  const getClientUrl = (client) => {
    return `http://localhost:${client.port}`
  }

  const checkClientStatus = async (client) => {
    try {
      const response = await axios.get(`${getClientUrl(client)}/api/status`, {
        timeout: 5000
      })
      
      const index = clients.value.findIndex(c => c.id === client.id)
      if (index !== -1) {
        clients.value[index] = {
          ...clients.value[index],
          ...response.data,
          isOnline: true,
          lastUpdate: new Date()
        }
      }
      return true
    } catch (error) {
      const index = clients.value.findIndex(c => c.id === client.id)
      if (index !== -1) {
        clients.value[index] = {
          ...clients.value[index],
          isOnline: false,
          lastUpdate: new Date()
        }
      }
      return false
    }
  }

  const checkAllStatus = async () => {
    isLoading.value = true
    
    await Promise.all(clients.value.map(client => checkClientStatus(client)))
    
    await checkServerStatus()
    
    isLoading.value = false
  }

  const checkServerStatus = async () => {
    try {
      const response = await axios.get(`${SERVER_URL}/api/status`, {
        timeout: 5000
      })
      serverStatus.value = {
        ...response.data,
        isOnline: true,
        lastUpdate: new Date()
      }
      return true
    } catch (error) {
      serverStatus.value = {
        isOnline: false,
        lastUpdate: new Date()
      }
      return false
    }
  }

  const fetchClientLosses = async (client) => {
    try {
      const response = await axios.get(`${getClientUrl(client)}/api/losses`, {
        timeout: 5000
      })
      
      if (response.data && response.data.loss_updates) {
        lossHistory.value[client.id] = response.data.loss_updates
      }
    } catch (error) {
      console.error(`Failed to fetch losses for ${client.id}:`, error)
    }
  }

  const fetchAllLosses = async () => {
    await Promise.all(clients.value.map(client => fetchClientLosses(client)))
  }

  const startTraining = async (learningRate, rounds) => {
    isLoading.value = true
    
    const results = await Promise.all(
      clients.value.filter(c => c.isOnline).map(async (client) => {
        try {
          const response = await axios.post(
            `${getClientUrl(client)}/api/train`,
            {
              learning_rate: learningRate,
              rounds: rounds
            },
            { timeout: 10000 }
          )
          return { clientId: client.id, success: true, data: response.data }
        } catch (error) {
          return { clientId: client.id, success: false, error: error.message }
        }
      })
    )
    
    isLoading.value = false
    return results
  }

  const stopTraining = async () => {
    isLoading.value = true
    
    const results = await Promise.all(
      clients.value.filter(c => c.isOnline).map(async (client) => {
        try {
          const response = await axios.post(
            `${getClientUrl(client)}/api/stop`,
            {},
            { timeout: 5000 }
          )
          return { clientId: client.id, success: true, data: response.data }
        } catch (error) {
          return { clientId: client.id, success: false, error: error.message }
        }
      })
    )
    
    isLoading.value = false
    return results
  }

  const resetClients = async () => {
    isLoading.value = true
    
    const results = await Promise.all(
      clients.value.filter(c => c.isOnline).map(async (client) => {
        try {
          await axios.post(`${getClientUrl(client)}/api/reset`, {}, { timeout: 5000 })
          lossHistory.value[client.id] = []
          return { clientId: client.id, success: true }
        } catch (error) {
          return { clientId: client.id, success: false, error: error.message }
        }
      })
    )
    
    await axios.post(`${SERVER_URL}/api/reset`, {}, { timeout: 5000 }).catch(() => {})
    
    isLoading.value = false
    return results
  }

  const aggregateWeights = async () => {
    try {
      const response = await axios.post(`${SERVER_URL}/api/aggregate`, {}, { timeout: 30000 })
      return response.data
    } catch (error) {
      throw error
    }
  }

  const startPolling = (interval = 2000) => {
    if (pollInterval.value) {
      stopPolling()
    }
    
    pollInterval.value = setInterval(async () => {
      await checkAllStatus()
      await fetchAllLosses()
    }, interval)
  }

  const stopPolling = () => {
    if (pollInterval.value) {
      clearInterval(pollInterval.value)
      pollInterval.value = null
    }
  }

  const onlineClients = computed(() => {
    return clients.value.filter(c => c.isOnline)
  })

  const trainingClients = computed(() => {
    return clients.value.filter(c => c.isOnline && c.is_training)
  })

  const allLossData = computed(() => {
    const data = {}
    
    for (const client of clients.value) {
      const losses = lossHistory.value[client.id] || []
      if (losses.length > 0) {
        data[client.id] = losses.map(l => ({
          x: l.epoch + (l.round - 1) * 5,
          y: l.loss,
          round: l.round,
          epoch: l.epoch
        }))
      }
    }
    
    return data
  })

  const getChartDatasets = computed(() => {
    const colors = [
      { border: '#00d4ff', background: 'rgba(0, 212, 255, 0.1)' },
      { border: '#00b894', background: 'rgba(0, 184, 148, 0.1)' },
      { border: '#e17055', background: 'rgba(225, 112, 85, 0.1)' }
    ]
    
    const datasets = []
    
    clients.value.forEach((client, index) => {
      const data = allLossData.value[client.id]
      if (data && data.length > 0) {
        const color = colors[index % colors.length]
        datasets.push({
          label: client.name,
          data: data,
          borderColor: color.border,
          backgroundColor: color.background,
          borderWidth: 2,
          tension: 0.3,
          fill: false,
          pointRadius: 3,
          pointHoverRadius: 5
        })
      }
    })
    
    return datasets
  })

  return {
    clients,
    lossHistory,
    isLoading,
    serverStatus,
    onlineClients,
    trainingClients,
    allLossData,
    getChartDatasets,
    init,
    checkAllStatus,
    checkClientStatus,
    checkServerStatus,
    fetchAllLosses,
    startTraining,
    stopTraining,
    resetClients,
    aggregateWeights,
    startPolling,
    stopPolling,
    SERVER_URL
  }
})
