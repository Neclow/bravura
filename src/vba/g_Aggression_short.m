function [gx] = g_Aggression_short(x_t, P, u_t, inG)
% Observation function for the aggression model.
%
% Computes predicted shock probability via a sigmoid link:
%   gx = sigmoid(Kp + Kr1*ShockedTm1 + Krc*ShockedSum - Kwc*WinSum)
%
% Parameters (phi):
%   Kp  — baseline proactive aggression
%   Kr1 — weight on provocation at t-1
%   Krc — weight on cumulative provocations
%   Kwc — weight on cumulative wins (hardcoded minus sign: positive Kwc = shock more when losing)
%
% Inputs (u):
%   ShockedTm1 — opponent shocked on previous trial (binary)
%   ShockedSum — cumulative opponent shocks (normalised)
%   WinSum     — cumulative wins (normalised)

[~, phi, u] = getStateParamInput(x_t, P, u_t, inG);

gx = phi.Kp + phi.Kr1 * u.ShockedTm1 + phi.Krc * u.ShockedSum - phi.Kwc * u.WinSum;
gx = 1 / (1 + exp(-gx));

end
