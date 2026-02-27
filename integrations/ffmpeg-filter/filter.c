/*
 * ea_threshold — decode video with FFmpeg, apply Eä threshold kernel,
 *                write raw grayscale output.
 *
 * This is NOT an in-tree AVFilter. It demonstrates the realistic embed
 * pattern: use libav* for decode/format, call an Eä kernel for compute.
 *
 * Usage: ./ea_threshold input.mp4 output.raw 128
 *        ffplay -f rawvideo -pix_fmt gray -video_size WxH output.raw
 */

#include <stdio.h>
#include <stdlib.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

/* Eä kernel — compiled separately, linked at build time */
extern void threshold_u8(const uint8_t *src, uint8_t *dst, int32_t n, uint8_t thresh);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <input.mp4> <output.raw> <threshold 0-255>\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    int thresh = atoi(argv[3]);
    if (thresh < 0 || thresh > 255) {
        fprintf(stderr, "threshold must be 0-255\n");
        return 1;
    }

    /* Open input */
    AVFormatContext *fmt_ctx = NULL;
    if (avformat_open_input(&fmt_ctx, input_path, NULL, NULL) < 0) {
        fprintf(stderr, "cannot open %s\n", input_path);
        return 1;
    }
    avformat_find_stream_info(fmt_ctx, NULL);

    /* Find video stream */
    int video_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = (int)i;
            break;
        }
    }
    if (video_idx < 0) {
        fprintf(stderr, "no video stream found\n");
        return 1;
    }

    AVCodecParameters *codecpar = fmt_ctx->streams[video_idx]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    avcodec_open2(codec_ctx, codec, NULL);

    int width  = codec_ctx->width;
    int height = codec_ctx->height;
    fprintf(stderr, "video: %dx%d, threshold: %d\n", width, height, thresh);

    /* Set up grayscale converter */
    struct SwsContext *sws = sws_getContext(
        width, height, codec_ctx->pix_fmt,
        width, height, AV_PIX_FMT_GRAY8,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    /* Allocate grayscale frame */
    AVFrame *frame = av_frame_alloc();
    AVFrame *gray  = av_frame_alloc();
    gray->format = AV_PIX_FMT_GRAY8;
    gray->width  = width;
    gray->height = height;
    av_frame_get_buffer(gray, 0);

    uint8_t *threshold_buf = malloc(width);
    FILE *out_file = fopen(output_path, "wb");
    if (!out_file) {
        fprintf(stderr, "cannot open %s for writing\n", output_path);
        return 1;
    }

    /* Decode, convert, threshold, write */
    AVPacket *pkt = av_packet_alloc();
    int frame_count = 0;

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        avcodec_send_packet(codec_ctx, pkt);
        while (avcodec_receive_frame(codec_ctx, frame) == 0) {
            /* Convert to grayscale */
            sws_scale(sws,
                (const uint8_t *const *)frame->data, frame->linesize,
                0, height,
                gray->data, gray->linesize
            );

            /* Apply Eä threshold kernel per row */
            for (int row = 0; row < height; row++) {
                uint8_t *src_row = gray->data[0] + row * gray->linesize[0];
                threshold_u8(src_row, threshold_buf, width, (uint8_t)thresh);
                fwrite(threshold_buf, 1, width, out_file);
            }
            frame_count++;
        }
        av_packet_unref(pkt);
    }

    fprintf(stderr, "processed %d frames -> %s\n", frame_count, output_path);
    fprintf(stderr, "play with: ffplay -f rawvideo -pix_fmt gray -video_size %dx%d %s\n",
            width, height, output_path);

    /* Cleanup */
    fclose(out_file);
    free(threshold_buf);
    av_packet_free(&pkt);
    av_frame_free(&gray);
    av_frame_free(&frame);
    sws_freeContext(sws);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return 0;
}
