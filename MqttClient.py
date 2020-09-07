import paho.mqtt.client as mqtt
import ssl

# The callback for when the client receives a CONNACK response from the server.
def onConnect(client, userdata, flags, rc):
    rcList = {
        0: "Connection successful",
        1: "Connection refused - incorrect protocol version",
        2: "Connection refused - invalid client identifier",
        3: "Connection refused - server unavailable",
        4: "Connection refused - bad username or password",
        5: "Connection refused",
    }
    print(rcList.get(rc, "Unknown server connection return code {}.".format(rc)))

# The callback for when a PUBLISH message is received from the server.
def onMessage(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

# Send message to SAP MQTT Server
def sendMessage(client, deviceID, messageContentJson):
    client.publish(deviceID, messageContentJson)    

def startMqttClient(deviceId, pemCertFilePath, mqttServerUrl, mqttServerPort, ackTopicLevel, sapIotDeviceID):
    client = mqtt.Client(deviceId) 
    client.on_connect = onConnect
    client.on_message = onMessage
    client.tls_set(certfile=pemCertFilePath, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS, ciphers=None)
    client.connect(mqttServerUrl, mqttServerPort)
    client.subscribe(ackTopicLevel+sapIotDeviceID) #Subscribe to device ack topic (feedback given from SAP IoT MQTT Server)
    client.loop_start() #Listening loop start
    return client