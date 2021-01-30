//TODO
//    отправка id сессии для проверки прав доступа   

var wsCmd = null;

function getStreamsConfig() {
    if (wsCmd != null) {
        if (wsCmd.readyState == 1) {
            wsCmd.send(JSON.stringify({ "cmd": "getStreamsConfig" }));
        } else {
            console.log("Websocket is not ready. Wait a sec.")
            setTimeout(() => { getStreamsConfig(); }, 500);
        }
    } else { console.log("Error websocket does not exist!!"); }
}

function getGpuList() {
    if (wsCmd != null) {
        if (wsCmd.readyState == 1) {
            wsCmd.send(JSON.stringify({ "cmd": "getCameras" }));
        } else {
            console.log("Websocket is not ready. Wait a sec.")
            setTimeout(() => { getCamerasList(); }, 500);
        }
    } else { console.log("Error websocket does not exist!!"); }
}

function getManagerData() {
    if (wsCmd != null) {
        if (wsCmd.readyState == 1) {
            wsCmd.send(JSON.stringify({ "cmd": "getManagerData" }));
        } else {
            console.log("Websocket is not ready. Wait a sec.")
            setTimeout(() => { getManagerData(); }, 500);
        }
    } else { console.log("Error websocket does not exist!!"); }
}

function WebSocketCmd() {
            
            if ("WebSocket" in window) {
               console.log("WebSocket is supported by your Browser!");
               // Let us open a web socket
               wsCmd = new WebSocket("ws://localhost:8080/wsCmd");
				
               wsCmd.onopen = function() {
                  
                  // Web Socket is connected, send data using send()
                  //wsCmd.send("Message!!!!");
                  console.log("WebSocket started!");
               };
				
               wsCmd.onmessage = function (evt) { 
                  // var received_msg = evt.data;
                  // console.log("Message is received!");
                   try {
                       if (evt.data instanceof Blob) {
                           console.log("Binary data!", evt.data);
                           showFrame(evt.data);
                       } else {
                          let jsonData = JSON.parse(evt.data);
                           // console.log("Message 2 is received: ", jsonData);
                           for (var item in jsonData) {
                               switch (item) {
                                   case 'streamsConfigList': {
                                       console.log("cameras list updating...");
                                       showToaster("<span style='color:blue'><b>Success!!!</b><br>Cameras list updated!!! </span>");

                                       localStorage.setItem('streamsConfigList', JSON.stringify(jsonData['streamsConfigList']));
                                       configViewUpdate('streamsConfigList');
                                       configViewUpdate('camsList');
                                           break;
                                       }
                                   case 'OK': {
                                       console.log("OK!", jsonData);
                                       showToaster("<span style='color:blue'><b>Success!!!</b><br>" + jsonData["OK"][0] + "</span>");

                                       // showToaster("Server said OK for " + jsonData["OK"][0]);
                                       updateViewOk(jsonData);
                                       break;
                                   }
                                   case 'error': {
                                       showToaster("<span style='color:red'><b>Error!</b><br>" + jsonData["error"][0]+"</span>");
                                       console.log("error!", jsonData);
                                       break;
                                   }
                                   case 'camsList': {
                                       // console.log("!!!managerConfig data updating...", jsonData);
                                       localStorage.setItem('camsList', JSON.stringify(jsonData['camsList']));
                                       configViewUpdate('camsList');
                                       break;
                                   }
                                   case 'managerConfig': {
                                       console.log("!!!managerConfig data updating...", jsonData);
                                       localStorage.setItem('managerConfig', JSON.stringify(jsonData['managerConfig']));
                                       configViewUpdate('managerConfig');
                                       break;
                                   }
                                   default: { // added brackets
                                       console.log('Empty action received.');
                                       break;
                                   }
                               }
                           }                          
                              // if (jsonData.hasOwnProperty('streamsConfigList')) {
                              //     console.log("cameras list updating...");
                              //     localStorage.setItem('streamsConfigList', JSON.stringify(jsonData['streamsConfigList']));
                              //     configViewUpdate('streamsConfigList');                                 
                              // } else if (jsonData.hasOwnProperty('gpusConfigList')) {
                              //         console.log("cameras list updating...");
                              //     localStorage.setItem('gpusConfigList', JSON.stringify(jsonData['gpusConfigList']));
                              //     configViewUpdate('gpusConfigList');                                 
                              // } else if (jsonData.hasOwnProperty('managerConfig')) {
                              //    console.log("!!!managerConfig data updating...", jsonData);
                              //    localStorage.setItem('managerConfig', JSON.stringify(jsonData['managerConfig']));
                              //     configViewUpdate('managerConfig');                                 
                              // } else if (jsonData.hasOwnProperty('camsList')) {
                              //     console.log("!!!managerConfig data updating...", jsonData);
                              //     localStorage.setItem('camsList', JSON.stringify(jsonData['camsList']));
                              //     configViewUpdate('camsList');
                              // } else if (jsonData.hasOwnProperty('camera')) {
                              //     console.log("!!!camera data updating...", jsonData);
                              // } else if (jsonData.hasOwnProperty('OK')) {
                              //    console.log("Server says OK", jsonData);
                              //} else {
                              //    console.log("Server can not detect command!!", evt.data);
                              //     }
                           // }
                      }
                  }catch (error) {
                        console.error(error);
                  }
               };

				
               wsCmd.onclose = function() {   
                  // websocket is closed.

                  console.log("Connection is closed..."); 
               };
            } else {
              
               // The browser doesn't support WebSocket
               console.log("WebSocket NOT supported by your Browser!");
            }
         }