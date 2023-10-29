---
title: '[Note] Super resolution cho ảnh truyền qua mạng lora network và cách cài đặt nvidia để inferences dựa vào pretrained model'
date: 2023-10-05
permalink: /posts/2023/10/05/super-resolution-cho-anh-truyen-qua-mang-lora-network-va-cach-cai-dat-nvidia-de-inferences-dua-vao-pretrained-model/
tags:
  - research
  - writing
  - nvidia
  - resolution
---

Trình bày cơ chế hoạt động của camera esp32 và cách thức để truyền dữ liệu camera qua mạng lora. Hơn nữa cách sử dụng nvidia để inferences cho bài toán super resolution.

Viết code sử dụng esp32-cam và deploy lên webserver
======

## Một số lưu ý 

Một số lưu ý khi viết code để thực hiện việc sử dụng esp32-cam và deploy lên webserver

* Cài đặt extension esp32 idf trong vscode
* Ctrl shift P để show các gợi ý cần tìm kiếm.
* Kết nối thông tin cổng COM của máy tính

Các bước thực hiện

* Create example project --> sample project
* Sau đó copy paste các đoạn code dưới 
* Clean projject
* Build project
* flash code
* Monitor device (lưu ý nhấn nút trên thiết bị esp32-cam)
* Sau đó hiển thị url để kết nối xem cam realtime
* Lưu ý: nếu có lỗi lầm thì thường rút thiết bị cắm lại. Kiểm tra thư viện và phiên bản cài đặt.

## Cấu trúc của một project

### 1. File camera_pins.h sẽ code như sau:

```c
#ifndef CAMERA_PINS_H_
#define CAMERA_PINS_H_

//#define CONFIG_BOARD_ESP32CAM_AITHINKER 1

// Freenove ESP32-WROVER CAM Board PIN Map
#if CONFIG_BOARD_WROVER_KIT
#define CAM_PIN_PWDN -1  //power down is not used
#define CAM_PIN_RESET -1 //software reset will be performed
#define CAM_PIN_XCLK 21
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27

#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 19
#define CAM_PIN_D2 18
#define CAM_PIN_D1 5
#define CAM_PIN_D0 4
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22
#endif

// ESP-EYE PIN Map
#if CONFIG_BOARD_CAMERA_MODEL_ESP_EYE
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    4
#define SIOD_GPIO_NUM    18
#define SIOC_GPIO_NUM    23

#define Y9_GPIO_NUM      36
#define Y8_GPIO_NUM      37
#define Y7_GPIO_NUM      38
#define Y6_GPIO_NUM      39
#define Y5_GPIO_NUM      35
#define Y4_GPIO_NUM      14
#define Y3_GPIO_NUM      13
#define Y2_GPIO_NUM      34
#define VSYNC_GPIO_NUM   5
#define HREF_GPIO_NUM    27
#define PCLK_GPIO_NUM    25
#endif

// AiThinker ESP32Cam PIN Map
#if CONFIG_BOARD_ESP32CAM_AITHINKER
#define CAM_PIN_PWDN 32
#define CAM_PIN_RESET -1 //software reset will be performed
#define CAM_PIN_XCLK 0
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27

#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 21
#define CAM_PIN_D2 19
#define CAM_PIN_D1 18
#define CAM_PIN_D0 5
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22
#endif

// TTGO T-Journal ESP32 Camera PIN Map
#if CONFIG_BOARD_CAMERA_MODEL_TTGO_T_JOURNAL
#define PWDN_GPIO_NUM      0
#define RESET_GPIO_NUM    15
#define XCLK_GPIO_NUM     27
#define SIOD_GPIO_NUM     25
#define SIOC_GPIO_NUM     23

#define Y9_GPIO_NUM       19
#define Y8_GPIO_NUM       36
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       39
#define Y5_GPIO_NUM        5
#define Y4_GPIO_NUM       34
#define Y3_GPIO_NUM       35
#define Y2_GPIO_NUM       17
#define VSYNC_GPIO_NUM    22
#define HREF_GPIO_NUM     26
#define PCLK_GPIO_NUM     21
#endif

#endif
```

### 2. File connect_wifi.h

```c
#ifndef CONNECT_WIFI_H_
#define CONNECT_WIFI_H_

#include <esp_system.h>
#include <nvs_flash.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "driver/gpio.h"
#include <lwip/sockets.h>
#include <lwip/sys.h>
#include <lwip/api.h>
#include <lwip/netdb.h>

extern int wifi_connect_status;

void connect_wifi(void);

#endif
```

### 3. File connect_wifi.c

```c
#include "connect_wifi.h"

int wifi_connect_status = 0;
static const char *TAG = "Connect_WiFi";
int s_retry_num = 0;


#define WIFI_SSID "Do Phuc Hao"
#define WIFI_PASSWORD "01234567890"
#define MAXIMUM_RETRY 5
/* FreeRTOS event group to signal when we are connected*/
EventGroupHandle_t s_wifi_event_group;

/* The event group allows multiple bits for each event, but we only care about two events:
 * - we are connected to the AP with an IP
 * - we failed to connect after the maximum amount of retries */
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        if (s_retry_num < MAXIMUM_RETRY)
        {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "retry to connect to the AP");
        }
        else
        {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
        wifi_connect_status = 0;
        ESP_LOGI(TAG, "connect to the AP fail");
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        wifi_connect_status = 1;
    }
}

void connect_wifi(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            /* Setting a password implies station will connect to all security modes including WEP/WPA.
             * However these modes are deprecated and not advisable to be used. Incase your Access point
             * doesn't support WPA2, these mode can be enabled by commenting below line */
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");

    /* Waiting until either the connection is established (WIFI_CONNECTED_BIT) or connection failed for the maximum
     * number of re-tries (WIFI_FAIL_BIT). The bits are set by event_handler() (see above) */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);

    /* xEventGroupWaitBits() returns the bits before the call returned, hence we can test which event actually
     * happened. */
    if (bits & WIFI_CONNECTED_BIT)
    {
        ESP_LOGI(TAG, "connected to ap SSID:%s password:%s",
                 WIFI_SSID, WIFI_PASSWORD);
    }
    else if (bits & WIFI_FAIL_BIT)
    {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s, password:%s",
                 WIFI_SSID, WIFI_PASSWORD);
    }
    else
    {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }
    vEventGroupDelete(s_wifi_event_group);
}
```

### 4. Hàm main.c

```c
#include <esp_system.h>
#include <nvs_flash.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "camera_pins.h"
#include "connect_wifi.h"

static const char *TAG = "esp32-cam Webserver";

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

#define CONFIG_XCLK_FREQ 20000000 

static esp_err_t init_camera(void)
{
    camera_config_t camera_config = {
        .pin_pwdn  = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,

        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,

        .xclk_freq_hz = CONFIG_XCLK_FREQ,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_VGA,

        .jpeg_quality = 10,
        .fb_count = 1,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY};//CAMERA_GRAB_LATEST. Sets when buffers should be filled
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        return err;
    }
    return ESP_OK;
}

esp_err_t jpg_stream_httpd_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len;
    uint8_t * _jpg_buf;
    char * part_buf[64];
    static int64_t last_frame = 0;
    if(!last_frame) {
        last_frame = esp_timer_get_time();
    }

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK){
        return res;
    }

    while(true){
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            break;
        }
        if(fb->format != PIXFORMAT_JPEG){
            bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
            if(!jpeg_converted){
                ESP_LOGE(TAG, "JPEG compression failed");
                esp_camera_fb_return(fb);
                res = ESP_FAIL;
            }
        } else {
            _jpg_buf_len = fb->len;
            _jpg_buf = fb->buf;
        }

        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if(res == ESP_OK){
            size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);

            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }
        if(fb->format != PIXFORMAT_JPEG){
            free(_jpg_buf);
        }
        esp_camera_fb_return(fb);
        if(res != ESP_OK){
            break;
        }
        int64_t fr_end = esp_timer_get_time();
        int64_t frame_time = fr_end - last_frame;
        last_frame = fr_end;
        frame_time /= 1000;
        /*
        ESP_LOGI(TAG, "MJPG: %uKB %ums (%.1ffps)",
            (uint32_t)(_jpg_buf_len/1024),
            (uint32_t)frame_time, 1000.0 / (uint32_t)frame_time);*/
    }

    last_frame = 0;
    return res;
}

httpd_uri_t uri_get = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = jpg_stream_httpd_handler,
    .user_ctx = NULL};
httpd_handle_t setup_server(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t stream_httpd  = NULL;

    if (httpd_start(&stream_httpd , &config) == ESP_OK)
    {
        httpd_register_uri_handler(stream_httpd , &uri_get);
    }

    return stream_httpd;
}

void app_main()
{
    esp_err_t err;

    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    connect_wifi();

    if (wifi_connect_status)
    {
        err = init_camera();
        if (err != ESP_OK)
        {
            printf("err: %s\n", esp_err_to_name(err));
            return;
        }
        setup_server();
        ESP_LOGI(TAG, "ESP32 CAM Web Server is up and running\n");
    }
    else
        ESP_LOGI(TAG, "Failed to connected with Wi-Fi, check your network Credentials\n");
}
```

### Note 

* Sau khi flash code thành công. Thì có thể chọn monitor device để xem output của nó. (nhớ nhấn nút trong thiết bị)

* Link để xem camera trực tiếp sẽ hiển thị ra.


Hướng dẫn làm việc với lora sx1276 và esp32-cam sử dụng esp32 idf extension vscode
======


## Một số lưu ý khi thực hiện

* Flash code trực tiếp vào thiết bị

* Khi flash thì nối esp32-cam với uart thì bình thường (nguồn với nguồn, gnd với gnd, tx và rx cũng tương ứng, còn GPIO0 vs GND của eps32-cam thì phải nối với nhau, còn khi sử dụng thì remove cái này đi) [cái này khi chỉ build mình với esp32-cam chứ chưa có lora.

* Khi có lora thì nối như sau: (Đây là node lora - có nghĩa là nó sẽ thu thập dữ liệu từ camera và gửi đi --> SENDER)

**Việc flash code xuống esp32-cam thì bình thường (như lưu ý ở trên)**
**Đầu tiên thì UART sẽ nối nguồn (5V, GND) trực tiếp với 5V và GND của esp32-cam.**
**Thứ hai, phần 3V, GND của esp32-cam sẽ nối với VCC và GND của lora sx1276**
**Thứ ba, phần tx, rx của esp32-cam sẽ nối trực tiếp với lora.**
**Thứ tư, cần anten vào tương ứng.**

* Đối với lora gateway (chỗ nhận - RECEIVER) thì nối cổng bình thường (5V, GND, TX, RX) với UART tương ứng.


## Phần code của camera

```c
/* UART asynchronous example, that uses separate RX and TX tasks

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "driver/uart.h"
#include "string.h"
#include "driver/gpio.h"

#include "esp_camera.h"
/**
 * @brief Data structure of camera frame buffer
 */
// typedef struct {
//     uint8_t * buf;              /*!< Pointer to the pixel data */
//     size_t len;                 /*!< Length of the buffer in bytes */
//     size_t width;               /*!< Width of the buffer in pixels */
//     size_t height;              /*!< Height of the buffer in pixels */
//     pixformat_t format;         /*!< Format of the pixel data */
//     struct timeval timestamp;   /*!< Timestamp since boot of the first DMA buffer of the frame */
// } camera_fb_t;

//ESP32Cam (AiThinker) PIN Map
#define CAM_PIN_PWDN 32
#define CAM_PIN_RESET -1 //software reset will be performed
#define CAM_PIN_XCLK 0
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27

#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 21
#define CAM_PIN_D2 19
#define CAM_PIN_D1 18
#define CAM_PIN_D0 5
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22

//static const char *TAG = "ESP32Cam";

static camera_config_t camera_config = {
    .pin_pwdn  = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sscb_sda = CAM_PIN_SIOD,
    .pin_sscb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    //XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,//YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_QQVGA,//QQVGA-QXGA Do not use sizes above QVGA when not JPEG

    .jpeg_quality = 50, //0-63 lower number means higher quality
    .fb_count = 1 //if more than one, i2s runs in continuous mode. Use only with JPEG
};

esp_err_t init_camera(){
    //power up the camera if PWDN pin is defined
    if(CAM_PIN_PWDN != -1){
    	gpio_set_direction(GPIO_NUM_32, GPIO_MODE_OUTPUT);
        // pinMode(CAM_PIN_PWDN, OUTPUT);
        gpio_set_level(GPIO_NUM_32, 0);
    }

    //initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        //ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}

esp_err_t camera_capture(){
    //acquire a frame
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        //ESP_LOGE(TAG, "Camera Capture Failed");
        //return ESP_FAIL; 
        return -1;
    }

    //replace this with your own function
    //ESP_LOGI(TAG, "Picture taken! Its width was: %zu pixels", fb->width);
    // ESP_LOGI(TAG, "Picture taken! Its height was: %zu pixels", fb->height);
    // ESP_LOGI(TAG, "Picture taken! Its format was: %zu bytes", fb->format);
    // ESP_LOGI(TAG, "Picture taken! Its buf was: %zu bytes", fb->buf);
    // ESP_LOGI(TAG, "Picture taken! Its len was: %zu bytes", fb->len);
  
    //return the frame buffer back to the driver for reuse
    esp_camera_fb_return(fb);
    return ESP_OK;
}

static const int RX_BUF_SIZE = 1024;

#define TXD_PIN (GPIO_NUM_1)
#define RXD_PIN (GPIO_NUM_3)

void init(void) {
    const uart_config_t uart_config = {
        .baud_rate = 9600,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_EVEN,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    // We won't use a buffer for sending data.
    uart_driver_install(UART_NUM_0, RX_BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uart_config);
    uart_set_pin(UART_NUM_0, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

int sendData(const char* logName, const char* data)
{
    const int len = strlen(data);
    const int txBytes = uart_write_bytes(UART_NUM_0, data, len);
    //ESP_LOGI(logName, "Wrote %d bytes", txBytes);
    return txBytes;
}

int sendGeneralFrame(uint8_t frameType, int8_t cmdType, uint8_t frameLoadLen, uint8_t* frameLoad)
{
    uint8_t checksum = 0;
    checksum ^= frameType;
    checksum ^= 0; // frameNum which is unused and always 0
    checksum ^= cmdType;
    checksum ^= frameLoadLen;

    uint8_t *frameByte = (uint8_t *) malloc((uint8_t) frameLoadLen + 5);
    frameByte[0] = frameType;
    frameByte[1] = (uint8_t) 0; // Frame Number = 0 by default.
    frameByte[2] = cmdType;
    frameByte[3] = frameLoadLen;
    memcpy(frameByte + sizeof(uint8_t) * 4, frameLoad, frameLoadLen);

    for (size_t i = 0; i < frameLoadLen; i++)
    {
        checksum ^= frameLoad[i];
    }
    frameByte[(uint8_t) frameLoadLen + 5 - 1] = checksum;

    uart_write_bytes(UART_NUM_0, (const char *) frameByte, (uint8_t) frameLoadLen + 5);
    
    free(frameByte);
    return 1;
}

int sendAppDataRequest( uint16_t target, uint8_t dataLen, uint8_t* data)
{
    // We add 7 bytes to the head of data for this payload
    uint8_t frameLoadLen = 6 + dataLen;
    uint8_t* frameLoad = (uint8_t *) malloc(sizeof(uint8_t) * frameLoadLen);

    // target address as big endian short
    frameLoad[0] = (uint8_t) ((target >> 8) & 0xFF);
    frameLoad[1] = (uint8_t) (target & 0xFF);

    // ACK request == 1 -> require acknowledgement of recv
    frameLoad[2] = (uint8_t) 0;

    // Send radius: which defaults to max of 7 hops, we can use that
    frameLoad[3] = (uint8_t) 7;

    // Discovery routing params == 1 -> automatic routing
    frameLoad[4] = (uint8_t) 1;

    // Data length
    frameLoad[5] = dataLen;

    // Data from index 7 to the end should be the data
    memcpy(frameLoad + (sizeof(uint8_t) * 6), data, dataLen);

    // frameType = 0x05, cmdType = 0x01 for sendData
    sendGeneralFrame(0x05, 0x01, frameLoadLen, frameLoad);
    
    free(frameLoad);

    return 1;
}

int testingSend()
{
    uint8_t imageID = 0;
    uint8_t seqNum = 0;
    for (int8_t i = 0; i < 5; i++) {
        uint8_t* data = malloc(sizeof(uint8_t) * 4);
        data[0] = (uint8_t) imageID;
        data[1] = (uint8_t) seqNum; // sequence Number - Fragment Number
        data[2] = 0x03;
        data[3] = 0x04;
        sendAppDataRequest(0x0000, 4, data);
        
        vTaskDelay(5000 / portTICK_PERIOD_MS);
        free(data);
        seqNum = seqNum + 1;
    }
    return 1;
}

void sendImage()
{
    //acquire a frame
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        uint8_t *nextACK = (uint8_t *) malloc(1);
        nextACK[0] = 0x31;
        uart_write_bytes(UART_NUM_0, (char *) nextACK, 1);
        free(nextACK);
        //ESP_LOGE(TAG, "Camera Capture Failed");
        //return ESP_FAIL;
        return;
    }
    //replace this with your own function
    //ESP_LOGI(TAG, "Picture taken! Its width was: %zu pixels", fb->width);
    //ESP_LOGI(TAG, "Picture taken! Its height was: %zu pixels", fb->height);
    // ESP_LOGI(TAG, "Picture taken! Its format was: %zu bytes", fb->format);
    // ESP_LOGI(TAG, "Picture taken! Its buf was: %zu bytes", fb->buf);
    //ESP_LOGI(TAG, "Picture taken! Its len was: %zu bytes", fb->len);

    //uint8_t* imageData = fb->buf;
    size_t imageDataLen = fb->len;
    // imageDataLen - Image Size in Byte
    size_t numOfFragment = 0;                  // Number of Fragments
    size_t fragmentSize = 100;                 // each fragment has 100 Bytes.
    size_t imageID = 0;

    if (imageDataLen % fragmentSize == 0)
        numOfFragment = imageDataLen / fragmentSize;
    else
        numOfFragment = imageDataLen / fragmentSize + 1;
    
    for (size_t i = 0; i < numOfFragment; i++) {
        // Sending each fragment
        if (i == numOfFragment - 1)
            fragmentSize = imageDataLen - (numOfFragment-1)*100;
        
        uint8_t* data = malloc(sizeof(uint8_t) * (fragmentSize + 2));
        data[0] = (uint8_t) imageID;     // Image ID
        data[1] = (uint8_t) i;     // Sequence Number

        for (size_t j = 0; j < fragmentSize; j++) {
            *(data+j+2) = *(fb->buf + j + i*fragmentSize);
        }

        //for (size_t j = 0; j < 100; j++) {
            // Data from index 3 to the end should be the image fragment data.
        //    memcpy(data + (sizeof(uint8_t) * (2+j)), &imageData[j + i*100], 1);
        //}
        sendAppDataRequest(0x0000, fragmentSize + 2, data);
        /*
        uint8_t* data = malloc(sizeof(uint8_t) * 4);
        data[0] = (uint8_t) imageID;
        data[1] = (uint8_t) i; // sequence Number - Fragment Number
        data[2] = 0x03;
        data[3] = 0x04;
        sendAppDataRequest(0x0000, 4, data);*/
        free(data);
        vTaskDelay(5000 / portTICK_PERIOD_MS);

    }
    
    //return the frame buffer back to the driver for reuse
    esp_camera_fb_return(fb);

}

void sendPacket()
{
    uint8_t *dataByte = (uint8_t *) malloc(16);
    dataByte[0] = 0x05;
    dataByte[1] = (uint8_t) 0;
    dataByte[2] = 0x01;
    dataByte[3] = 0x0b;
    dataByte[4] = 0x02;
    dataByte[5] = (uint8_t) 0;
    dataByte[6] = (uint8_t) 0;
    dataByte[7] = 0x07;
    dataByte[8] = 0x01;
    dataByte[9] = 0x05;
    dataByte[10] = 0x48;
    dataByte[11] = 0x65;
    dataByte[12] = 0x6c;
    dataByte[13] = 0x6c;
    dataByte[14] = 0x6f;
    dataByte[15] = 0x4c;
    uart_write_bytes(UART_NUM_0, (const char *) dataByte, 16);
    free(dataByte);
}

static void tx_task(void *arg)
{
    //static const char *TX_TASK_TAG = "TX_TASK";
    //esp_log_level_set(TX_TASK_TAG, ESP_LOG_INFO);
    while (1) {
        // sendData(TX_TASK_TAG, "Hello world");
        //char data[] = {0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64};
        //char data[] = {0x05, 0, 0x01, 0x0b, 0x02, 0, 0, 0x07, 0x01, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x4c};
        //sendData(TX_TASK_TAG, data);

        //testingSend();
        sendImage();

        vTaskDelay(20000 / portTICK_PERIOD_MS);
    }
}

/*
static void rx_task(void *arg)
{
    //static const char *RX_TASK_TAG = "RX_TASK";
    //esp_log_level_set(RX_TASK_TAG, ESP_LOG_INFO);
    uint8_t* data = (uint8_t*) malloc(RX_BUF_SIZE+1);
    while (1) {
        const int rxBytes = uart_read_bytes(UART_NUM_0, data, RX_BUF_SIZE, 1000 / portTICK_RATE_MS);
        if (rxBytes > 0) {
            data[rxBytes] = 0;
            //ESP_LOGI(RX_TASK_TAG, "Read %d bytes: '%s'", rxBytes, data);
            //ESP_LOG_BUFFER_HEXDUMP(RX_TASK_TAG, data, rxBytes, ESP_LOG_INFO);
        }
    }
    free(data);
}
*/
void app_main(void)
{
    init_camera();
    init();
    //xTaskCreate(rx_task, "uart_rx_task", 1024*2, NULL, configMAX_PRIORITIES, NULL);
    xTaskCreate(tx_task, "uart_tx_task", 1024*2, NULL, configMAX_PRIORITIES-1, NULL);
    //sendImage();
    //testingSend();
}

```

## Phần code của receiver (code python đọc từ cổng serial)

### Lưu ý khi thực hiện code

* Cài đặt thư viện pyserial

* Phần code chỗ port ở ubuntu có thể sửa thành: 

```sh
port='/dev/ttyUSB0',\
```

* Chú ý: Khi run port serial thì run ở lora gate (chỗ nhận) --> cổng của UART của lora gate. Khi đang run, thì cổng UART của nó sẽ bật đèn thể hiện việc đang running.

```python
import serial
import time
from os.path import join, dirname, realpath
import os
import time
import threading

port = serial.Serial(
    port = 'COM40',\
    baudrate=9600,\
    parity = serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
    timeout=0
)

print("Connected to: " + port.port)

portBuffer = []

recvTime = 0
numPacket = 0
imgNum = 0

rand_name = 'image_' + str(time.time()) + '.jpg'
path = join(join(dirname(realpath(__file__)), 'data'), rand_name)
f = open(path, 'wb')

while True:
    try:
        dataShow = []
        while port.inWaiting() > 0:
            recvTime = time.time()
            readByte = port.read()
            portBuffer.append(readByte)
            dataShow.append(readByte)
            if len(dataShow) > 20:
                #print("> ")
                #print(dataShow)
                dataShow = []
        time.sleep(1)

        if (time.time() - recvTime > 4) and len(portBuffer) != 0:
            rand_name = 'image_' + str(time.time()) + '.jpg'
            #rand_name = 'image_' + '.jpg'

            #print("Новое изображение было получено!\n")
            #print("Lenght of portBuffer = %d" % len(portBuffer))
            
            imageBuffer = []
            imgNum += 1
            dataPath = join(join(dirname(realpath(__file__)), 'data'), rand_name)
            print('len portBuffer: ', len(portBuffer))
            if f.closed:
                print('open file')
                f = open(dataPath, 'wb')
            for i in range(0, len(portBuffer)):
                if i < len(portBuffer) - 5:
                    if portBuffer[i] == b'\x05' and portBuffer[i+1] == b'\x00' and portBuffer[i+2] == b'\x82':
                        numPacket += 1
                        for j in range(0, int.from_bytes(portBuffer[i+7], 'big') - 2):
                            imageBuffer.append(portBuffer[i+10+j])
                            f.write(portBuffer[i+10+j])

            print("Number of packets = %d" % numPacket)
            numPacket = 0
            print("Image size = %d" % len(imageBuffer))
            if len(imageBuffer) < 100:
                print('closed file')
                f.close()
            portBuffer = []

    except KeyboardInterrupt:
        print("Good bye!!")
        break

```


Hướng dẫn làm việc với lora sx1276 và esp32-cam sử dụng esp32 idf extension vscode
======


## Một số lưu ý khi triển khai cài đặt với nvidia

### Flash OS

Một số lưu ý khi thực hiện 

* Xác định loại nvidia mà mình đang dùng. Mình đã mua nvidia dạng này: [Link avermedia](https://www.avermedia.com/professional/products?category=Carrier-Board)

* Chọn version nên được cài đặt phù hợp với thiết bị của bạn: [Link version](https://www.avermedia.com/professional/product-detail/NX215)

* Link: [Link download](https://s3.us-west-2.amazonaws.com/storage.avermedia.com/web_release_www/NX215B/BSP/2022-03-16/NX215B-R1.0.11.4.6.zip)

Các bước thực hiện cài đặt cho nvidia

* Bước 1: Download file ở trên

```sh
Để giải nén file dạng tar.gz thì gõ lệnh sau:

sudo tar zxf file.tar.gz

Vào trong folder: JetPack/Linux_for_Tegra 

Gõ: sudo ./setup.sh

Chọn default: rasberry 2
```

* Bước 2: Chuẩn bị SD card

```sh
Initialize the SD card and create new ext4 partition.

export sdcard=/dev/sdb

sudo gdisk $sdcard

Link:  https://www.youtube.com/watch?v=kjptoLT7nck

```

* Bước 3: Connext NX to Host

```sh
Cái này cần phải kết nối từ từ và cẩn thận.

Nếu gặp lỗi: string not found
sudo apt install binutils

Hoặc lỗi ascii 0x2a...
sudo apt install python-is-python3
```

### Cài đặt Cuda

* Tham khảo phần video hướng dẫn ở sau: 
**https://youtu.be/LUxyNyCl4ro**
**https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048**


```sh
sudo apt install nvidia-jetpack
```

If get error then:

```sh
apt remove
apt remove bluez
apt upgrade
apt install bluez
```

### Install pytorch and torchvision

```sh
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whlsudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

```sh
$ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev$ git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
$ cd torchvision
$ export BUILD_VERSION=0.9.0  # where 0.x.0 is the torchvision version  
$ python3 setup.py install --user
$ cd ../  # attempting to load torchvision from build dir will result in import error
$ pip install 'pillow<9'
```

Để verify

```sh
>>> import torch>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> print('cuDNN version: ' + str(torch.backends.cudnn.version()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
>>> import torchvision
>>> print(torchvision.version)
```

### GAN restore image

* Link pretrained model: [Link](https://github.com/cszn/BSRGAN/tree/5ce1a9c6ae292f30ccfce4b597ecb73c70401733)

### Type of file

* Install package: [Link package](https://pypi.org/project/filetype/)

### Code Lora Gateway and high resolution

```python
import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net


"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""

testsets = 'testsets'
testset_H = 'H_Image_Lora'
model_names = ['BSRGAN']
save_results = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getModel():

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    #testsets = 'testsets'       # fixed, set path of testsets
    #testset_H = 'H_Image_Lora'  # ['RealSRSet','DPED']

    #model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    #model_names = ['BSRGAN']    # 'BSRGANx2' for scale factor 2



    #save_results = True
    sf = 4
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth')          # set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()
        
        return model
        
        '''
        for testset_L in testset_Ls:

            L_path = os.path.join(testsets, testset_L)
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf))
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0

            for img in util.get_image_paths(L_path):
                print('test print image')
                print(img)

                # --------------------------------
                # (1) img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)

                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                if save_results:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))
        '''

import sys

def main(img):
    #print('main')
    #img = '/home/nvidia/BSRGAN/testsets/L_Image_Lora/Lincoln.png'
    try:

        OutPath = os.path.join(testsets, testset_H)
        img_name, ext = os.path.splitext(os.path.basename(img))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        #model = getModel()
    
        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)
    
        util.imsave(img_E, os.path.join(OutPath, img_name+'_' + 'H' + '.png'))
        print('written')
    except Interrupt:
        return

def RUN():
    import serial
    import time
    from os.path import join, dirname, realpath
    import os
    import threading
    import filetype


    port = serial.Serial(
        port='/dev/ttyUSB0',\
        baudrate=9600,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=0
    )

    print('Connected to : ' + port.port)

    portBuffer = []

    recvTime = 0
    numPacket = 0
    imgNum = 0

    rand_name = 'image_' + str(time.time()) + '.jpg'
    path = join(join(dirname(realpath(__file__)), 'testsets/L_Image_Lora'), rand_name)

    f = open(path, 'wb')
    
    while True:
        try:
            dataShow = []
            while port.inWaiting() > 0:
                recvTime = time.time()
                readByte = port.read()
                portBuffer.append(readByte)
                dataShow.append(readByte)
                if len(dataShow) > 20:
                    dataShow = []
            time.sleep(1)

            if time.time() - recvTime > 4 and len(portBuffer) != 0:
                rand_name = 'image_' + str(time.time()) + '.jpg'
                imageBuffer = []
                imgNum += 1
                dataPath = join(join(dirname(realpath(__file__)), 'testsets/L_Image_Lora'), rand_name)
                print('len portBuffer: ', len(portBuffer))

                if f.closed:
                    print('closed')
                    f = open(dataPath, 'wb')
                for i in range(0, len(portBuffer)):
                    if i < len(portBuffer) - 5:
                        if portBuffer[i] == b'\x05' and portBuffer[i+1] == b'\x00' and portBuffer[i+2] == b'\x82':
                            numPacket += 1
                            for j in range(0, int.from_bytes(portBuffer[i+7], 'big') - 2):
                                imageBuffer.append(portBuffer[i+10+j])
                                f.write(portBuffer[i+10+j])
                print('Number of packet = %d' % numPacket)
                numPacket = 0
                print('image size = %d' % len(imageBuffer))
                if len(imageBuffer) < 100:
                    print('closed file')
                    nameFile = f.name
                    f.close()

                    kind = filetype.guess(nameFile)
                    if kind is None:
                        print('failed')
                    else:
                        print(kind.mime)
                        print('call high resolution module')
                        main(nameFile)

                portBuffer = []

        except KeyBoardInterrupt:
            print('Good bye')
            break

if __name__ == '__main__':
    param1 = '/home/nvidia/BSRGAN/testsets/L_Image_Lora/Lincoln.png'

    if len(sys.argv) > 1:
        param1 = sys.argv[1]
    
    model = getModel()

    #main(param1)

    RUN()
```


Hết.
