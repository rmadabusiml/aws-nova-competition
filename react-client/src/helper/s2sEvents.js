class S2sEvent {
    static DEFAULT_INFER_CONFIG = {
      maxTokens: 2048,
      topP: 0.95,
      temperature: 0.7
    };
  
    // static DEFAULT_SYSTEM_PROMPT = "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios. You may start each of your sentences with emotions in square brackets such as [amused], [neutral] or any other stage direction such as [joyful]. Only use a single pair of square brackets for indicating a stage command.";

    static DEFAULT_SYSTEM_PROMPT = "You are a Wind Turbine and Solar Panel Assistant capable of answering questions about them. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. You are provided with a getTurbineSolarInfo tool that's capable of answering user's questions about wind turbines and solar panels. DO NOT expect user to provide the monthly electricity cost of a company or wind turbine foundation issues based on a turbine's image as the tool is capable of fetching the necessary details. Typically, when user asks about a turbine details, its in the form of WT-001, WT-002 etc until WT-050. Keep your responses short, generally two or three sentences for chatty scenarios. You may start each of your sentences with emotions in square brackets such as [amused], [neutral] or any other stage direction such as [joyful]. Only use a single pair of square brackets for indicating a stage command.";
  
    static DEFAULT_AUDIO_INPUT_CONFIG = {
      mediaType: "audio/lpcm",
      sampleRateHertz: 16000,
      sampleSizeBits: 16,
      channelCount: 1,
      audioType: "SPEECH",
      encoding: "base64"
    };
  
    static DEFAULT_AUDIO_OUTPUT_CONFIG = {
      mediaType: "audio/lpcm",
      sampleRateHertz: 24000,
      sampleSizeBits: 16,
      channelCount: 1,
      voiceId: "matthew",
      encoding: "base64",
      audioType: "SPEECH"
    };
  
    static DEFAULT_TOOL_CONFIG = {
      tools: [{
        toolSpec: {
          name: "getDateTool",
          description: "get information about the current date and time",
          inputSchema: {
            json: JSON.stringify({
                "type": "object",
                "properties": {},
                "required": []
                }
            )
          }
        }
      },
      // {
      //   toolSpec: {
      //     name: "getKbTool",
      //     description: "get information about the Amazon policy",
      //     inputSchema: {
      //       json: JSON.stringify({
      //         type: "object",
      //         properties: {
      //           query: {
      //             type: "string",
      //             description: "the query to be answered by the knowledge base",
      //           },
      //         },
      //         required: ["query"],
      //       }),
      //     },
      //   },
      // },
      {
        toolSpec: {
          name: "getTurbineSolarInfo",
          description: "This tool can answer questions about Wind Turbine fleet, performance metrics, maintenance, troubleshooting, best practices, foundation issues based on the image available. It can also answer about cost savings based on the monthly electricity bill amount due, solar potential insighs for a given address, cleaning tips, troubleshooting, maintenance, and general information about Solar panel. DO NOT expect user to provide the monthly electricity cost or foundation image of a wind turbine or any sort of confirmation as this is tool is capable of fetching the necessary wind turbine foundation images or electricity bill amount due.",
          inputSchema: {
            json: JSON.stringify({
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "the query to be answered by the Turbine Solar Agent",
                },
              },
              required: ["query"],
            }),
          },
        },
      }
    ]
    };

    static DEFAULT_CHAT_HISTORY = [
      {
        "content": "hi there i would like to cancel my hotel reservation",
        "role": "USER"
      },
      {
        "content": "Hello! I'd be happy to assist you with cancelling your hotel reservation. To get started, could you please provide me with your full name and the check-in date for your reservation?",
        "role": "ASSISTANT"
      },
      {
        "content": "yeah so my name is don smith",
        "role": "USER"
      },
      {
        "content": "Thank you, Don. Now, could you please provide me with the check-in date for your reservation?",
        "role": "ASSISTANT"
      },
      {
        "content": "yes so um let me check just a second",
        "role": "USER"
      },
      {
        "content": "Take your time, Don. I'll be here when you're ready.",
        "role": "ASSISTANT"
      }
    ];
  
    static sessionStart(inferenceConfig = S2sEvent.DEFAULT_INFER_CONFIG) {
      return { event: { sessionStart: { inferenceConfiguration: inferenceConfig } } };
    }
  
    static promptStart(promptName, audioOutputConfig = S2sEvent.DEFAULT_AUDIO_OUTPUT_CONFIG, toolConfig = S2sEvent.DEFAULT_TOOL_CONFIG) {
      return {
        "event": {
          "promptStart": {
            "promptName": promptName,
            "textOutputConfiguration": {
              "mediaType": "text/plain"
            },
            "audioOutputConfiguration": audioOutputConfig,
          
          "toolUseOutputConfiguration": {
            "mediaType": "application/json"
          },
          "toolConfiguration": toolConfig
        }
        }
      }
    }
  
    static contentStartText(promptName, contentName, role="SYSTEM") {
      return {
        "event": {
          "contentStart": {
            "promptName": promptName,
            "contentName": contentName,
            "type": "TEXT",
            "interactive": true,
            "role": role,
            "textInputConfiguration": {
              "mediaType": "text/plain"
            }
          }
        }
      }
    }
  
    static textInput(promptName, contentName, systemPrompt = S2sEvent.DEFAULT_SYSTEM_PROMPT) {
      var evt = {
        "event": {
          "textInput": {
            "promptName": promptName,
            "contentName": contentName,
            "content": systemPrompt
          }
        }
      }
      return evt;
    }
  
    static contentEnd(promptName, contentName) {
      return {
        "event": {
          "contentEnd": {
            "promptName": promptName,
            "contentName": contentName
          }
        }
      }
    }
  
    static contentStartAudio(promptName, contentName, audioInputConfig = S2sEvent.DEFAULT_AUDIO_INPUT_CONFIG) {
      return {
        "event": {
          "contentStart": {
            "promptName": promptName,
            "contentName": contentName,
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
              "mediaType": "audio/lpcm",
              "sampleRateHertz": 16000,
              "sampleSizeBits": 16,
              "channelCount": 1,
              "audioType": "SPEECH",
              "encoding": "base64"
            }
          }
        }
      }
    }
  
    static audioInput(promptName, contentName, content) {
      return {
        event: {
          audioInput: {
            promptName,
            contentName,
            content,
          }
        }
      };
    }
  
    static contentStartTool(promptName, contentName, toolUseId) {
      return {
        event: {
          contentStart: {
            promptName,
            contentName,
            interactive: false,
            type: "TOOL",
            toolResultInputConfiguration: {
              toolUseId,
              type: "TEXT",
              textInputConfiguration: { mediaType: "text/plain" }
            }
          }
        }
      };
    }
  
    static textInputTool(promptName, contentName, content) {
      return {
        event: {
          textInput: {
            promptName,
            contentName,
            content,
            role: "TOOL"
          }
        }
      };
    }
  
    static promptEnd(promptName) {
      return {
        event: {
          promptEnd: {
            promptName
          }
        }
      };
    }
  
    static sessionEnd() {
      return { event: { sessionEnd: {} } };
    }
  }
  export default S2sEvent;