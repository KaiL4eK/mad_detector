# Possible <object_class>:
#     forward
#     forward_left
#     forward_right
#     left
#     right
#     brick
#     traffic_light_green
#     traffic_light_red
string object_class
# In range [0; 1]
float64 probability
# In range [0; 1]
geometry_msgs/Point ul_point
# In range [0; 1]
geometry_msgs/Point br_point
# In range [0; pic_sz]
geometry_msgs/Point size_px
