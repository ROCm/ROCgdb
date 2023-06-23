# Copyright 2022, 2023 Free Software Foundation, Inc.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import gdb
import gdb.printing

# This is deprecated in 3.9, but required in older versions.
from typing import Optional

from .frames import frame_for_id
from .server import capability, request, client_bool_capability
from .startup import send_gdb_with_response, in_gdb_thread
from .varref import find_variable, VariableReference


class EvaluateResult(VariableReference):
    def __init__(self, value):
        super().__init__(None, value, "result")


# Helper function to evaluate an expression in a certain frame.
@in_gdb_thread
def _evaluate(expr, frame_id):
    global_context = True
    if frame_id is not None:
        frame = frame_for_id(frame_id)
        frame.select()
        global_context = False
    val = gdb.parse_and_eval(expr, global_context=global_context)
    ref = EvaluateResult(val)
    return ref.to_object()


# Like _evaluate but ensure that the expression cannot cause side
# effects.
@in_gdb_thread
def _eval_for_hover(expr, frame_id):
    with gdb.with_parameter("may-write-registers", "off"):
        with gdb.with_parameter("may-write-memory", "off"):
            with gdb.with_parameter("may-call-functions", "off"):
                return _evaluate(expr, frame_id)


# Helper function to perform an assignment.
@in_gdb_thread
def _set_expression(expression, value, frame_id):
    global_context = True
    if frame_id is not None:
        frame = frame_for_id(frame_id)
        frame.select()
        global_context = False
    lhs = gdb.parse_and_eval(expression, global_context=global_context)
    rhs = gdb.parse_and_eval(value, global_context=global_context)
    lhs.assign(rhs)
    return EvaluateResult(lhs).to_object()


# Helper function to evaluate a gdb command in a certain frame.
@in_gdb_thread
def _repl(command, frame_id):
    if frame_id is not None:
        frame = frame_for_id(frame_id)
        frame.select()
    val = gdb.execute(command, from_tty=True, to_string=True)
    return {
        "result": val,
        "variablesReference": 0,
    }


@request("evaluate")
@capability("supportsEvaluateForHovers")
def eval_request(
    *,
    expression: str,
    frameId: Optional[int] = None,
    context: str = "variables",
    **args,
):
    if context in ("watch", "variables"):
        # These seem to be expression-like.
        return send_gdb_with_response(lambda: _evaluate(expression, frameId))
    elif context == "hover":
        return send_gdb_with_response(lambda: _eval_for_hover(expression, frameId))
    elif context == "repl":
        return send_gdb_with_response(lambda: _repl(expression, frameId))
    else:
        raise Exception('unknown evaluate context "' + context + '"')


@in_gdb_thread
def _variables(ref, start, count):
    var = find_variable(ref)
    children = var.fetch_children(start, count)
    return [x.to_object() for x in children]


@request("variables")
# Note that we ignore the 'filter' field.  That seems to be
# specific to javascript.
def variables(*, variablesReference: int, start: int = 0, count: int = 0, **args):
    # This behavior was clarified here:
    # https://github.com/microsoft/debug-adapter-protocol/pull/394
    if not client_bool_capability("supportsVariablePaging"):
        start = 0
        count = 0
    result = send_gdb_with_response(
        lambda: _variables(variablesReference, start, count)
    )
    return {"variables": result}


@capability("supportsSetExpression")
@request("setExpression")
def set_expression(
    *, expression: str, value: str, frameId: Optional[int] = None, **args
):
    return send_gdb_with_response(lambda: _set_expression(expression, value, frameId))
