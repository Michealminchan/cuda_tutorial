add_lab("ThrustVectorAdd")
add_lab_template("ThrustVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("ThrustVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("ThrustVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
