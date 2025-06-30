## Use the command to generate grpc packages
```shell
python -m grpc_tools.protoc -I ./ ./*.proto --python_out=./ --grpc_python_out=./
```