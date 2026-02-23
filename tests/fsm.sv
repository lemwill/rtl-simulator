module fsm (
    input  logic       clk,
    input  logic       rst,
    input  logic       go,
    output logic [1:0] state,
    output logic       done
);
    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam WAIT = 2'd2;
    localparam DONE_ST = 2'd3;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (go)
                        state <= RUN;
                end
                RUN:
                    state <= WAIT;
                WAIT:
                    state <= DONE_ST;
                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default:
                    state <= IDLE;
            endcase
        end
    end
endmodule
