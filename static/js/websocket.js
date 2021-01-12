//TODO
//    отправка id сессии для проверки прав доступа   

var wsCmd = null;
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
                  console.log("Message is received!");
                   try {
                       if (evt.data instanceof Blob) {
                           //console.log("Binary data!");
                           putFrame(evt.data);
                       } else {
                          let jsonData = JSON.parse(evt.data);
                          console.log("Message is received: ", jsonData);
                           if (jsonData.hasOwnProperty('streamsConfigList')) {
                               console.log("cameras list updating...");
                               localStorage.setItem('streamsConfigList', JSON.stringify(jsonData['streamsConfigList']));
                               configViewUpdate('streamsConfigList');                                 
                           } else if (jsonData.hasOwnProperty('gpusConfigList')) {
                                   console.log("cameras list updating...");
                               localStorage.setItem('gpusConfigList', JSON.stringify(jsonData['gpusConfigList']));
                               configViewUpdate('gpusConfigList');                                 
                           } else if (jsonData.hasOwnProperty('managerConfig')) {
                              console.log("!!!managerConfig data updating...", jsonData);
                              localStorage.setItem('managerConfig', JSON.stringify(jsonData['managerConfig']));
                               configViewUpdate('managerConfig');                                 
                          } else if (jsonData.hasOwnProperty('camera')) {
                              console.log("!!!camera data updating...", jsonData);
                          } else {
                              console.log("Can not detect websocket command!!", evt.data);
                          }
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