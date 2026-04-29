<template>
  <div class="card">
    <h2>训练控制面板</h2>
    
    <div class="grid grid-2">
      <div>
        <h3>训练参数</h3>
        
        <div class="form-group">
          <label>学习率 (Learning Rate)</label>
          <input 
            type="number" 
            v-model.number="localConfig.learningRate" 
            step="0.001"
            min="0.0001"
            max="1"
          />
        </div>
        
        <div class="form-group">
          <label>训练轮数 (Rounds)</label>
          <input 
            type="number" 
            v-model.number="localConfig.rounds" 
            min="1"
            max="100"
          />
        </div>
        
        <div class="form-group">
          <label>每轮训练周期 (Epochs per Round)</label>
          <input 
            type="number" 
            v-model.number="localConfig.epochsPerRound" 
            min="1"
            max="50"
          />
        </div>
      </div>
      
      <div>
        <h3>操作控制</h3>
        
        <div class="btn-group" style="margin-bottom: 20px;">
          <button 
            class="btn btn-primary" 
            @click="handleStartTraining"
            :disabled="store.isLoading || store.trainingClients.length > 0"
          >
            <span v-if="store.isLoading" class="loading-spinner"></span>
            <span v-else>开始训练</span>
          </button>
          
          <button 
            class="btn btn-danger" 
            @click="handleStopTraining"
            :disabled="store.isLoading || store.trainingClients.length === 0"
          >
            停止训练
          </button>
          
          <button 
            class="btn btn-secondary" 
            @click="handleReset"
            :disabled="store.isLoading"
          >
            重置系统
          </button>
        </div>
        
        <div style="margin-bottom: 15px;">
          <button 
            class="btn btn-success" 
            @click="handleAggregate"
            :disabled="store.isLoading"
            style="width: 100%;"
          >
            手动触发权重聚合 (FedAvg)
          </button>
        </div>
        
        <div class="form-group">
          <label>自动刷新间隔 (ms)</label>
          <select v-model="pollInterval" @change="updatePolling">
            <option :value="1000">1 秒</option>
            <option :value="2000">2 秒 (默认)</option>
            <option :value="5000">5 秒</option>
            <option :value="0">关闭</option>
          </select>
        </div>
      </div>
    </div>
    
    <div v-if="message" class="notification" :class="messageType">
      {{ message }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch } from 'vue'
import { useClientsStore } from '../stores/clients'

const store = useClientsStore()

const localConfig = reactive({
  learningRate: 0.01,
  rounds: 10,
  epochsPerRound: 5
})

const pollInterval = ref(2000)
const message = ref('')
const messageType = ref('info')

const showMessage = (msg, type = 'info') => {
  message.value = msg
  messageType.value = type
  setTimeout(() => {
    message.value = ''
  }, 3000)
}

const handleStartTraining = async () => {
  try {
    const results = await store.startTraining(
      localConfig.learningRate,
      localConfig.rounds
    )
    
    const successCount = results.filter(r => r.success).length
    showMessage(`训练指令已发送至 ${successCount}/${store.onlineClients.length} 个客户端`, 'success')
  } catch (error) {
    showMessage('启动训练失败: ' + error.message, 'error')
  }
}

const handleStopTraining = async () => {
  try {
    await store.stopTraining()
    showMessage('停止训练指令已发送', 'info')
  } catch (error) {
    showMessage('停止训练失败: ' + error.message, 'error')
  }
}

const handleReset = async () => {
  try {
    await store.resetClients()
    showMessage('系统已重置', 'success')
  } catch (error) {
    showMessage('重置失败: ' + error.message, 'error')
  }
}

const handleAggregate = async () => {
  try {
    const result = await store.aggregateWeights()
    showMessage(`聚合完成，当前轮次: ${result.current_round}`, 'success')
  } catch (error) {
    showMessage('聚合失败: ' + error.message, 'error')
  }
}

const updatePolling = () => {
  if (pollInterval.value > 0) {
    store.startPolling(pollInterval.value)
  } else {
    store.stopPolling()
  }
}

updatePolling()
</script>

<style scoped>
.notification {
  padding: 12px 16px;
  border-radius: 8px;
  margin-top: 15px;
  font-weight: 500;
}

.notification.success {
  background: rgba(0, 184, 148, 0.2);
  color: #00b894;
}

.notification.error {
  background: rgba(231, 76, 60, 0.2);
  color: #e74c3c;
}

.notification.info {
  background: rgba(0, 212, 255, 0.2);
  color: #00d4ff;
}
</style>
