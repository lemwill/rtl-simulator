// SPI Master — exercises FSM, shift register, enum, case equality
module spi_master (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] result
);
    // States
    typedef enum logic [1:0] {
        IDLE  = 2'd0,
        LOAD  = 2'd1,
        SHIFT = 2'd2,
        DONE  = 2'd3
    } state_t;

    state_t state;
    logic [7:0]  shift_reg;
    logic [2:0]  bit_count;
    logic [7:0]  tx_data;
    logic        sclk;
    logic        mosi;

    // LFSR for self-stimulus
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hA5A5A5A5;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    assign tx_data = lfsr[7:0];

    // SPI FSM
    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            shift_reg <= 8'd0;
            bit_count <= 3'd0;
            sclk      <= 1'b0;
            mosi      <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (lfsr[8])
                        state <= LOAD;
                end
                LOAD: begin
                    shift_reg <= tx_data;
                    bit_count <= 3'd7;
                    state     <= SHIFT;
                end
                SHIFT: begin
                    sclk      <= ~sclk;
                    mosi      <= shift_reg[7];
                    shift_reg <= {shift_reg[6:0], 1'b0};
                    if (bit_count === 3'd0)
                        state <= DONE;
                    else
                        bit_count <= bit_count - 3'd1;
                end
                DONE: begin
                    sclk  <= 1'b0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    // Parity of transmitted byte using $countones
    logic [3:0] ones_count;
    assign ones_count = $countones(shift_reg);

    // Capture tx_data when loading
    logic [7:0] last_tx;
    always_ff @(posedge clk) begin
        if (rst)
            last_tx <= 8'd0;
        else if (state == LOAD)
            last_tx <= tx_data;
    end

    // Checksum accumulator — XOR when transmission completes
    logic [31:0] checksum;
    always_ff @(posedge clk) begin
        if (rst)
            checksum <= 32'd0;
        else if (state == DONE)
            checksum <= checksum ^ {24'd0, last_tx} ^ {28'd0, ones_count};
    end

    assign result = checksum;
endmodule
