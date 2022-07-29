# Makefile that contains several make targets for building 
# the Docker image and running the container.                                                                                                                                                                                                                           
                                                                                                                                                                                      
.PHONY: help

CONTAINER_NAME ?= news-classifier

help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
                                                                                                                                                                                                     
list-containers: ## List all containers
	docker container ls --all

build: ## Build the Docker image
	docker build --platform linux/amd64 -t news-classifier .

run: ## Run the container
	docker run --name $(CONTAINER_NAME) --rm -p 80:80 news-classifier

exec: # SSH into the container to check the logs.out file  
	docker exec -it $(CONTAINER_NAME) /bin/sh

stop: ## Stop the container
	docker stop $(CONTAINER_NAME)