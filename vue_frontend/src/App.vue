<template>
  <div id="app">
    <header class="header">
      <h1>FedAvg 联邦学习监控系统</h1>
      <p>
        分布式机器学习模拟沙盒 | 
        C++ 聚合服务器 + Python GNN 客户端 + Vue 3 前端控制台
      </p>
    </header>
    
    <ControlPanel />
    
    <ClientStatus />
    
    <LossChart />
    
    <div class="card">
      <h2>系统架构说明</h2>
      <div class="grid grid-3">
        <div class="client-card">
          <h4>🖥️ C++ 聚合服务器 (端口 8080)</h4>
          <p>• 接收多客户端权重张量</p>
          <p>• 执行 FedAvg 加权聚合</p>
          <p>• 管理训练轮次状态</p>
          <p>• 提供 REST API 接口</p>
        </div>
        
        <div class="client-card">
          <h4>🐍 Python 客户端 (端口 5001-5003)</h4>
          <p>• 加载隔离的 Non-IID 数据集</p>
          <p>• 运行 GCN 图神经网络训练</p>
          <p>• 上传本地权重到聚合服务器</p>
          <p>• 提供 API 接收前端指令</p>
        </div>
        
        <div class="client-card">
          <h4>🌐 Vue 3 前端 (端口 3000)</h4>
          <p>• 配置学习率和训练轮数</p>
          <p>• 向各节点下发训练指令</p>
          <p>• 实时监控客户端状态</p>
          <p>• 绘制 Loss 收敛曲线图</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import { useClientsStore } from './stores/clients'
import ControlPanel from './components/ControlPanel.vue'
import ClientStatus from './components/ClientStatus.vue'
import LossChart from './components/LossChart.vue'

const store = useClientsStore()

onMounted(async () => {
  await store.init()
  store.startPolling(2000)
})

onUnmounted(() => {
  store.stopPolling()
})
</script>

<style>
/* 全局样式已在 style.css 中定义 */
</style>
