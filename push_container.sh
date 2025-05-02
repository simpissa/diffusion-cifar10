#!/bin/bash
docker tag diffusion-cifar10 $DOCKER_USER/diffusion-cifar10:latest
docker push $DOCKER_USER/diffusion-cifar10:latest