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
                  console.log("Message is received: ", evt.data);
                  try{

                      let jsonData = JSON.parse(evt.data);
                      console.log("Message is received: ", jsonData);
                      if(jsonData.hasOwnProperty('cameras')){
                            console.log("cameras list updating...");
                            jsonData.cameras.forEach((row) => { addRow(row); });
                      }else if(jsonData.hasOwnProperty('camera')){
                        console.log("!!!camera data updating...", jsonData);
                      }else{
                        console.log("Can not detect websocket command!!", evt.data);
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