<template>
  <div class="card">
    <h2>客户端状态</h2>
    
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">总客户端</div>
        <div class="stat-value">{{ store.clients.length }}</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-label">在线</div>
        <div class="stat-value" style="color: #00b894;">{{ store.onlineClients.length }}</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-label">训练中</div>
        <div class="stat-value" style="color: #00d4ff;">{{ store.trainingClients.length }}</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-label">聚合服务器</div>
        <div class="stat-value">
          <span v-if="store.serverStatus?.isOnline" style="color: #00b894;">在线</span>
          <span v-else style="color: #e74c3c;">离线</span>
        </div>
      </div>
    </div>
    
    <div class="grid grid-3">
      <div v-for="client in store.clients" :key="client.id" class="client-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
          <h4>{{ client.name }}</h4>
          <div class="connection-status">
            <span class="dot" :class="client.isOnline ? 'dot-online' : 'dot-offline'"></span>
            <span style="font-size: 0.85rem;">{{ client.isOnline ? '在线' : '离线' }}</span>
          </div>
        </div>
        
        <div v-if="client.isOnline">
          <p>
            端口: <span>{{ client.port }}</span>
          </p>
          <p>
            训练样本: <span>{{ client.sample_count }}</span>
          </p>
          <p>
            当前状态: 
            <span class="status-badge" :class="client.is_training ? 'status-running' : 'status-idle'">
              {{ client.is_training ? '训练中' : '空闲' }}
            </span>
          </p>
          <p v-if="client.is_training">
            训练轮次: <span>{{ client.current_round }} / {{ client.max_rounds }}</span>
          </p>
          <p>
            学习率: <span>{{ client.lr }}</span>
          </p>
        </div>
        
        <div v-else>
          <p style="color: #e74c3c;">客户端未连接</p>
        </div>
      </div>
    </div>
    
    <div v-if="store.serverStatus" class="card" style="margin-top: 20px; background: #0f3460;">
      <h3>聚合服务器状态</h3>
      <div class="grid grid-2">
        <div>
          <p>地址: <span>{{ store.SERVER_URL }}</span></p>
          <p>当前轮次: <span>{{ store.serverStatus.current_round || '-' }}</span></p>
          <p>总轮次: <span>{{ store.serverStatus.max_rounds || '-' }}</span></p>
        </div>
        <div>
          <p>期望客户端数: <span>{{ store.serverStatus.expected_clients || '-' }}</span></p>
          <p>已提交客户端: <span>{{ store.serverStatus.submitted_clients || '-' }}</span></p>
          <p>本轮是否完成: <span>{{ store.serverStatus.is_complete ? '是' : '否' }}</span></p>
        </div>
      </div>
      
      <div v-if="store.serverStatus.clients?.length > 0" style="margin-top: 15px;">
        <h4>已提交权重的客户端:</h4>
        <div v-for="c in store.serverStatus.clients" :key="c.client_id" style="padding: 8px; background: #1a1a2e; border-radius: 6px; margin-bottom: 5px;">
          <span>{{ c.client_id }}</span> - 
          <span style="color: #00d4ff;">{{ c.sample_count }} 样本</span> - 
          <span>第 {{ c.round }} 轮</span>
        </div>
      </div>
    </div>
    
    <div style="margin-top: 20px; text-align: center;">
      <button class="btn btn-secondary" @click="refreshStatus" :disabled="store.isLoading">
        <span v-if="store.isLoading" class="loading-spinner"></span>
        <span v-else>刷新状态</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { useClientsStore } from '../stores/clients'

const store = useClientsStore()

const refreshStatus = async () => {
  await store.checkAllStatus()
}
</script>
