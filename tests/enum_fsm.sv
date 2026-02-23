// FSM with enum types — tests enum value resolution
module enum_fsm (
    input  logic       clk,
    input  logic       rst,
    output logic [1:0] state,
    output logic [7:0] count
);
    typedef enum logic [1:0] {
        IDLE  = 2'd0,
        RUN   = 2'd1,
        WAIT  = 2'd2,
        DONE  = 2'd3
    } state_t;

    state_t current;

    always_ff @(posedge clk) begin
        if (rst) begin
            current <= IDLE;
            count   <= 8'd0;
        end else begin
            case (current)
                IDLE: current <= RUN;
                RUN: begin
                    count <= count + 8'd1;
                    if (count == 8'd9)
                        current <= WAIT;
                end
                WAIT: current <= DONE;
                DONE: begin
                    current <= IDLE;
                    count   <= 8'd0;
                end
                default: current <= IDLE;
            endcase
        end
    end

    assign state = current;
endmodule
