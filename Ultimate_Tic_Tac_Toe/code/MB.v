
module led_matrix (
    input wire clk,
    input wire reset,
    input wire sel,
    input wire [3:0]p,
    input wire [17:0]Tout,
    output reg [5:0]row,
    output reg [11:0]col
);

    reg [2:0]row_sel;
    reg [11:0]pattern[5:0];
    reg [4:0]state;
    reg [15:0]counter;
    
    integer i , j;
    
    // Reset pattern on falling edge of sel
    // State machine to handle Tout bits in sequence
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < 6; i = i + 1) begin
                pattern[i] <= 12'd0;
            end
            row_sel <= 3'd0;
            state <= 0;
            counter <= 0;
        end 
        else if (sel) begin
            for (i = 0; i < 6; i = i + 1) begin
                pattern[i] <= 12'd0;
            end
            state <= 0;
        end
        else begin
            counter <= counter + 1;
            if(counter[15]) begin
                row_sel <= row_sel + 1;
                if(row_sel == 3'd5) begin
                    row_sel <= 3'd0;
                end
                counter <= 0;
            end   
            
            row <= 3'b001 << row_sel;
            col <= ~pattern[row_sel];
            
            case (state)
                0: begin
                    if (Tout[0] && !Tout[1]) begin
                        pattern[0] <= pattern[0] | 12'b010100000000;
                        pattern[1] <= pattern[1] | 12'b010000000000;
                    end
                    else if (Tout[1] && !Tout[0]) begin
                        pattern[0] <= pattern[0] | 12'b101000000000;
                        pattern[1] <= pattern[1] | 12'b100000000000;
                    end
                    else begin
                        pattern[0] <= pattern[0] & 12'b000011111111;
                        pattern[1] <= pattern[1] & 12'b000011111111;
                    end
                    state <= state + 1;
                end
                1: begin
                    if (Tout[2] && !Tout[3]) begin
                        pattern[0] <= pattern[0] | 12'b000001010000;
                        pattern[1] <= pattern[1] | 12'b000001000000;
                    end
                    else if (Tout[3] && !Tout[2]) begin
                        pattern[0] <= pattern[0] | 12'b000010100000;
                        pattern[1] <= pattern[1] | 12'b000010000000;
                    end
                    else begin
                        pattern[0] <= pattern[0] & 12'b111100001111;
                        pattern[1] <= pattern[1] & 12'b111100001111;
                    end
                    state <= state + 1;
                end
                2: begin
                    if (Tout[4] && !Tout[5]) begin
                        pattern[0] <= pattern[0] | 12'b000000000101;
                        pattern[1] <= pattern[1] | 12'b000000000100;
                    end
                    else if (Tout[5] && !Tout[4]) begin
                        pattern[0] <= pattern[0] | 12'b000000001010;
                        pattern[1] <= pattern[1] | 12'b000000001000;
                    end
                    else begin
                        pattern[0] <= pattern[0] & 12'b111111110000;
                        pattern[1] <= pattern[1] & 12'b111111110000;
                    end
                    state <= state + 1;
                end
                
                3: begin
                    if (Tout[6] && !Tout[7]) begin
                        pattern[2] <= pattern[2] | 12'b010100000000;
                        pattern[3] <= pattern[3] | 12'b010000000000;
                    end
                    else if (Tout[7] && !Tout[6]) begin
                        pattern[2] <= pattern[2] | 12'b101000000000;
                        pattern[3] <= pattern[3] | 12'b100000000000;
                    end
                    else begin
                        pattern[2] <= pattern[2] & 12'b000011111111;
                        pattern[3] <= pattern[3] & 12'b000011111111;
                    end
                    state <= state + 1;
                end
                4: begin
                    if (Tout[8] && !Tout[9]) begin
                        pattern[2] <= pattern[2] | 12'b000001010000;
                        pattern[3] <= pattern[3] | 12'b000001000000;
                    end
                    else if (Tout[9] && !Tout[8]) begin
                        pattern[2] <= pattern[2] | 12'b000010100000;
                        pattern[3] <= pattern[3] | 12'b000010000000;
                    end
                    else begin
                        pattern[2] <= pattern[2] & 12'b111100001111;
                        pattern[3] <= pattern[3] & 12'b111100001111;
                    end
                    state <= state + 1;
                end
                5: begin
                    if (Tout[10] && !Tout[11]) begin
                        pattern[2] <= pattern[2] | 12'b000000000101;
                        pattern[3] <= pattern[3] | 12'b000000000100;
                    end
                    else if (Tout[11] && !Tout[10]) begin
                        pattern[2] <= pattern[2] | 12'b000000001010;
                        pattern[3] <= pattern[3] | 12'b000000001000;
                    end
                    else begin
                        pattern[2] <= pattern[2] & 12'b111111110000;
                        pattern[3] <= pattern[3] & 12'b111111110000;
                    end
                    state <= state + 1;
                end
                
                6: begin
                    if (Tout[12] && !Tout[13]) begin
                        pattern[4] <= pattern[4] | 12'b010100000000;
                        pattern[5] <= pattern[5] | 12'b010000000000;
                    end
                    else if (Tout[13] && !Tout[12]) begin
                        pattern[4] <= pattern[4] | 12'b101000000000;
                        pattern[5] <= pattern[5] | 12'b100000000000;
                    end
                    else begin
                        pattern[4] <= pattern[4] & 12'b000011111111;
                        pattern[5] <= pattern[5] & 12'b000011111111;
                    end
                    state <= state + 1;
                end
                7: begin
                    if (Tout[14] && !Tout[15]) begin
                        pattern[4] <= pattern[4] | 12'b000001010000;
                        pattern[5] <= pattern[5] | 12'b000001000000;
                    end
                    else if (Tout[15] && !Tout[14]) begin
                        pattern[4] <= pattern[4] | 12'b000010100000;
                        pattern[5] <= pattern[5] | 12'b000010000000;
                    end
                    else begin
                        pattern[4] <= pattern[4] & 12'b111100001111;
                        pattern[5] <= pattern[5] & 12'b111100001111;
                    end
                    state <= state + 1;
                end
                8: begin
                    if (Tout[16] && !Tout[17]) begin
                        pattern[4] <= pattern[4] | 12'b000000000101;
                        pattern[5] <= pattern[5] | 12'b000000000100;
                    end
                    else if (Tout[17] && !Tout[16]) begin
                        pattern[4] <= pattern[4] | 12'b000000001010;
                        pattern[5] <= pattern[5] | 12'b000000001000;
                    end
                    else begin
                        pattern[4] <= pattern[4] & 12'b111111110000;
                        pattern[5] <= pattern[5] & 12'b111111110000;
                    end
                    state <= state + 1;
                end
                9: begin
                    if (p == 0) begin
                        pattern[1] <= (pattern[1] & 12'b110011001100) | 12'b000100000000;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= pattern[5] & 12'b110011001100;
                    end
                    state <= state + 1;
                end
                10: begin
                    if (p == 1) begin
                        pattern[1] <= (pattern[1] & 12'b110011001100) | 12'b000000010000;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= pattern[5] & 12'b110011001100;
                    end
                    state <= state + 1;
                end
                11: begin
                    if (p == 2) begin
                        pattern[1] <= (pattern[1] & 12'b110011001100) | 12'b000000000001;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= pattern[5] & 12'b110011001100;
                    end
                    state <= state + 1;
                end
                12: begin
                    if (p == 3) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= (pattern[3] & 12'b110011001100) | 12'b000100000000;
                        pattern[5] <= pattern[5] & 12'b110011001100;
                    end
                    state <= state + 1;
                end
                13: begin
                    if (p == 4) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= (pattern[3] & 12'b110011001100) | 12'b000000010000;
                        pattern[5] <= pattern[5] & 12'b110011001100;;
                    end
                    state <= state + 1;
                end
                14: begin
                    if (p == 5) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= (pattern[3] & 12'b110011001100) | 12'b000000000001;
                        pattern[5] <= pattern[5] & 12'b110011001100;
                    end
                    state <= state + 1;
                end
                15: begin
                    if (p == 6) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= (pattern[5] & 12'b110011001100) | 12'b000100000000;
                    end
                    state <= state + 1;
                end
                16: begin
                    if (p == 7) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= (pattern[5] & 12'b110011001100) | 12'b000000010000;
                    end
                    state <= state + 1;
                end
                17: begin
                    if (p == 8) begin
                        pattern[1] <= pattern[1] & 12'b110011001100;
                        pattern[3] <= pattern[3] & 12'b110011001100;
                        pattern[5] <= (pattern[5] & 12'b110011001100) | 12'b000000000001;
                    end
                    state <= 0;
                end
                default: state <= 0;
            endcase
        end
    end

endmodule