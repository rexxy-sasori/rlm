#!/bin/bash

# Build script for RLM Docker images
# Usage: ./build.sh <dockerfile_name> <image_tag>
# Example: ./build.sh Dockerfile.visualizer visualizer-v0.1.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_NAME="${1:-}"
IMAGE_TAG="${2:-}"
REGISTRY="harbor.xa.xshixun.com:7443/hanfeigeng/rlm"

# Show usage if arguments are missing
if [ -z "$DOCKERFILE_NAME" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <dockerfile_name> <image_tag>"
    echo ""
    echo "Examples:"
    echo "  $0 Dockerfile.visualizer visualizer-v0.1.0"
    echo "  $0 Dockerfile.rlm rlm-oolong-v0.1.2"
    echo ""
    echo "Available Dockerfiles in docker/:"
    ls -1 "$SCRIPT_DIR/docker/Dockerfile"* 2>/dev/null || echo "  (none found)"
    exit 1
fi

DOCKERFILE_PATH="$SCRIPT_DIR/docker/$DOCKERFILE_NAME"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE_PATH"
    exit 1
fi

echo "Building image..."
echo "  Dockerfile: $DOCKERFILE_PATH"
echo "  Tag: $REGISTRY:$IMAGE_TAG"
echo "  Platform: linux/amd64"
echo ""

docker buildx build \
    --platform=linux/amd64 \
    -f "$DOCKERFILE_PATH" \
    -t "$REGISTRY:$IMAGE_TAG" \
    "$SCRIPT_DIR"

echo ""
echo "Build complete: $REGISTRY:$IMAGE_TAG"
echo ""
echo "To push the image, run:"
echo "  docker push $REGISTRY:$IMAGE_TAG"
