<template>
  <div class="card">
    <h2>Loss 收敛曲线</h2>
    
    <div v-if="hasData" class="chart-container">
      <Line :data="chartData" :options="chartOptions" />
    </div>
    
    <div v-else class="empty-state">
      <p>暂无训练数据</p>
      <p class="subtext">开始训练后，Loss 曲线将实时显示</p>
    </div>
    
    <div v-if="hasData" style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #0f3460;">
      <h3>最近 Loss 值</h3>
      <div class="grid grid-3">
        <div v-for="(data, clientId) in store.allLossData" :key="clientId" class="client-card">
          <h4>{{ getClientName(clientId) }}</h4>
          <p>
            最新 Loss: 
            <span style="color: #00d4ff; font-weight: 600;">
              {{ getLatestLoss(clientId) }}
            </span>
          </p>
          <p>
            数据点数: <span>{{ data.length }}</span>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { useClientsStore } from '../stores/clients'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const store = useClientsStore()

const hasData = computed(() => {
  return Object.values(store.allLossData).some(d => d && d.length > 0)
})

const getClientName = (clientId) => {
  const client = store.clients.find(c => c.id === clientId)
  return client ? client.name : clientId
}

const getLatestLoss = (clientId) => {
  const data = store.allLossData[clientId]
  if (data && data.length > 0) {
    return data[data.length - 1].y.toFixed(4)
  }
  return '-'
}

const chartData = computed(() => {
  const datasets = store.getChartDatasets
  
  let allLabels = new Set()
  datasets.forEach(ds => {
    ds.data.forEach(d => {
      allLabels.add(d.x)
    })
  })
  
  const labels = Array.from(allLabels).sort((a, b) => a - b)
  
  return {
    labels: labels.map(String),
    datasets: datasets
  }
})

const chartOptions = ref({
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'index',
    intersect: false
  },
  plugins: {
    legend: {
      display: true,
      position: 'top',
      labels: {
        color: '#eaeaea',
        font: {
          size: 12
        }
      }
    },
    tooltip: {
      backgroundColor: 'rgba(15, 52, 96, 0.9)',
      titleColor: '#00d4ff',
      bodyColor: '#eaeaea',
      borderColor: '#00d4ff',
      borderWidth: 1,
      padding: 12,
      callbacks: {
        label: function(context) {
          return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`
        }
      }
    }
  },
  scales: {
    x: {
      display: true,
      title: {
        display: true,
        text: '训练步数 (Epoch)',
        color: '#a0a0a0'
      },
      grid: {
        color: 'rgba(160, 160, 160, 0.1)'
      },
      ticks: {
        color: '#a0a0a0'
      }
    },
    y: {
      display: true,
      title: {
        display: true,
        text: 'Loss 值',
        color: '#a0a0a0'
      },
      grid: {
        color: 'rgba(160, 160, 160, 0.1)'
      },
      ticks: {
        color: '#a0a0a0'
      }
    }
  }
})
</script>

<style scoped>
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #a0a0a0;
}

.empty-state p {
  font-size: 1.2rem;
  margin-bottom: 10px;
}

.empty-state .subtext {
  font-size: 0.9rem;
  color: #666;
}
</style>
