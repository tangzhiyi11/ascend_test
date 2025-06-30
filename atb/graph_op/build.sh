g++ -I "${ATB_HOME_PATH}/include" -D_GLIBCXX_USE_CXX11_ABI=0 -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" *.cpp -l atb -l ascendcl -o graph_op_demo
