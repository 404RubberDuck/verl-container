# verl-container

build the wheels from 


## Build container
```
podman build -t verl:0.4.1 .
```

## Compress into .sqsh on alps
```
enroot import -x mount -o verl-0.4.1.sqsh podman://verl:0.4.1
```
