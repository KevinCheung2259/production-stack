#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保KUBECONFIG已正确设置
if [ -z "${KUBECONFIG}" ]; then
  # 如果KUBECONFIG未设置，使用默认路径
  export KUBECONFIG=~/.kube/config
  echo "KUBECONFIG未设置，使用默认配置: ${KUBECONFIG}"
fi

# 验证是否可以连接到Kubernetes
if ! kubectl cluster-info &>/dev/null; then
  echo "错误: 无法连接到Kubernetes集群。请确保集群正在运行并且配置正确。"
  exit 1
fi

echo "成功连接到Kubernetes集群"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm upgrade --install kube-prom-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  -f kube-prom-stack.yaml --wait

helm install prometheus-adapter prometheus-community/prometheus-adapter \
    --namespace monitoring \
    -f "$SCRIPT_DIR/prom-adapter.yaml"
