// A realistic UART transmitter - common real-world design
module uart_tx #(
    parameter CLKS_PER_BIT = 87  // 115200 baud at 10MHz
)(
    input  logic       clk, rst,
    input  logic       tx_valid,
    input  logic [7:0] tx_data,
    output logic       tx_out,
    output logic       tx_busy,
    output logic       tx_done
);
    typedef enum logic [2:0] {
        IDLE   = 3'd0,
        START  = 3'd1,
        DATA   = 3'd2,
        STOP   = 3'd3,
        DONE   = 3'd4
    } state_t;

    state_t state;
    logic [$clog2(CLKS_PER_BIT)-1:0] clk_count;
    logic [2:0] bit_idx;
    logic [7:0] tx_shift;

    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            tx_out    <= 1'b1;  // idle high
            tx_busy   <= 1'b0;
            tx_done   <= 1'b0;
            clk_count <= '0;
            bit_idx   <= 3'd0;
            tx_shift  <= 8'd0;
        end else begin
            tx_done <= 1'b0;

            case (state)
                IDLE: begin
                    tx_out  <= 1'b1;
                    tx_busy <= 1'b0;
                    if (tx_valid) begin
                        state    <= START;
                        tx_shift <= tx_data;
                        tx_busy  <= 1'b1;
                    end
                end

                START: begin
                    tx_out <= 1'b0;  // start bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        state     <= DATA;
                        bit_idx   <= 3'd0;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DATA: begin
                    tx_out <= tx_shift[0];
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        tx_shift  <= {1'b0, tx_shift[7:1]};  // shift right
                        if (bit_idx == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                STOP: begin
                    tx_out <= 1'b1;  // stop bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        state     <= DONE;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DONE: begin
                    tx_done <= 1'b1;
                    tx_busy <= 1'b0;
                    state   <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule

// Top-level with self-stimulus
module test_uart(
    input  logic        clk, rst,
    output logic [31:0] result
);
    logic       tx_valid;
    logic [7:0] tx_data;
    logic       tx_out, tx_busy, tx_done;
    logic [31:0] counter;
    logic [7:0]  bytes_sent;

    uart_tx #(.CLKS_PER_BIT(8)) u_tx (
        .clk(clk), .rst(rst),
        .tx_valid(tx_valid), .tx_data(tx_data),
        .tx_out(tx_out), .tx_busy(tx_busy), .tx_done(tx_done)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            counter    <= 32'd0;
            bytes_sent <= 8'd0;
            tx_valid   <= 1'b0;
            tx_data    <= 8'd0;
        end else begin
            counter <= counter + 32'd1;
            tx_valid <= 1'b0;

            if (!tx_busy && counter[3:0] == 4'd0) begin
                tx_valid <= 1'b1;
                tx_data  <= counter[7:0];
            end

            if (tx_done)
                bytes_sent <= bytes_sent + 8'd1;
        end
    end

    assign result = {bytes_sent, 7'd0, tx_out, 7'd0, tx_busy, counter[7:0]};
endmodule
