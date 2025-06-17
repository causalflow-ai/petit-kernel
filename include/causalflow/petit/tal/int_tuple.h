/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "causalflow/petit/tal/algorithm/functional.h"
#include "causalflow/petit/tal/algorithm/tuple_algorithms.h"
#include "causalflow/petit/tal/config.h"
#include "causalflow/petit/tal/container/tuple.h"
#include "causalflow/petit/tal/numeric/integral_constant.h"

/** IntTuple is an integer or a tuple of IntTuples.
 * This file holds utilities for working with IntTuples,
 * but does not hold a concrete concept or class of IntTuple.
 */

namespace causalflow::tal {

// Implementation of get<0>(Integral).
//   Even though is_tuple<Integral> is false and tuple_size<Integral> doesn't
//   compile, CuTe defines rank(Integral) as 1, so it's useful for
//   get<0>(Integral) to return its input
template <size_t I, class T,
          __TAL_REQUIRES(std::is_integral<std::remove_cvref_t<T>>::value)>
TAL_HOST_DEVICE constexpr decltype(auto) get(T &&t) noexcept {
    static_assert(I == 0, "Index out of range");
    return static_cast<T &&>(t);
}

// Custom recursive get for anything that implements get<I>(.) (for a single
// integer I).
template <size_t I0, size_t I1, size_t... Is, class T>
TAL_HOST_DEVICE constexpr decltype(auto) get(T &&t) noexcept {
    return get<I1, Is...>(get<I0>(static_cast<T &&>(t)));
}

//
// shape
//

template <class IntTuple>
TAL_HOST_DEVICE constexpr auto shape(IntTuple const &s) {
    if constexpr (is_tuple<IntTuple>::value) {
        return transform(s, [](auto const &a) { return shape(a); });
    } else {
        return s;
    }

    unreachable();
}

template <int I, int... Is, class IntTuple>
TAL_HOST_DEVICE constexpr auto shape(IntTuple const &s) {
    if constexpr (is_tuple<IntTuple>::value) {
        return shape<Is...>(get<I>(s));
    } else {
        return get<I, Is...>(shape(s));
    }

    unreachable();
}

//
// max
//

template <class T0, class... Ts>
TAL_HOST_DEVICE constexpr auto max(T0 const &t0, Ts const &...ts) {
    if constexpr (is_tuple<T0>::value) {
        return tal::max(
            tal::apply(t0, [](auto const &...a) { return tal::max(a...); }),
            ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return tal::max(t0, tal::max(ts...));
    }

    unreachable();
}

//
// min
//

template <class T0, class... Ts>
TAL_HOST_DEVICE constexpr auto min(T0 const &t0, Ts const &...ts) {
    if constexpr (is_tuple<T0>::value) {
        return tal::min(
            tal::apply(t0, [](auto const &...a) { return tal::min(a...); }),
            ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return tal::min(t0, tal::min(ts...));
    }

    unreachable();
}

//
// gcd
//

template <class T0, class... Ts>
TAL_HOST_DEVICE constexpr auto gcd(T0 const &t0, Ts const &...ts) {
    if constexpr (is_tuple<T0>::value) {
        return tal::gcd(
            tal::apply(t0, [](auto const &...a) { return tal::gcd(a...); }),
            ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return tal::gcd(t0, tal::gcd(ts...));
    }

    unreachable();
}

//
// product
//

// Implementation of product as a function object
struct Product {
    template <class IntTuple>
    TAL_HOST_DEVICE constexpr auto operator()(IntTuple const &a) const {
        if constexpr (is_tuple<IntTuple>::value) {
            if constexpr (tuple_size<IntTuple>::value == 0) {
                return Int<1>{};
            } else {
                return tal::transform_apply(a, Product{},
                                            multiplies_unary_lfold{});
            }
        } else if constexpr (tal::is_integral<IntTuple>::value) {
            return a;
        }

        unreachable();
    }
};
// Callable product function object
TAL_INLINE_CONSTANT Product product;

// Return a rank(t) tuple @a result such that get<i>(@a result) =
// product(get<i>(@a t))
template <class Tuple>
TAL_HOST_DEVICE constexpr auto product_each(Tuple const &t) {
    return transform(wrap(t), product);
}

// Take the product of Tuple at the leaves of TupleG
template <class Tuple, class TupleG>
TAL_HOST_DEVICE constexpr auto product_like(Tuple const &tuple,
                                            TupleG const &guide) {
    return transform_leaf(
        guide, tuple, [](auto const &g, auto const &t) { return product(t); });
}

// Return the product of elements in a mode
template <int... Is, class IntTuple>
TAL_HOST_DEVICE constexpr auto size(IntTuple const &a) {
    if constexpr (sizeof...(Is) == 0) {
        return product(a);
    } else {
        return size(get<Is...>(a));
    }

    unreachable();
}

template <class IntTuple>
static constexpr auto size_v = decltype(size(declval<IntTuple>()))::value;

//
// sum
//

template <class IntTuple>
TAL_HOST_DEVICE constexpr auto sum(IntTuple const &a) {
    if constexpr (is_tuple<IntTuple>::value) {
        return tal::apply(
            a, [](auto const &...v) { return (Int<0>{} + ... + sum(v)); });
    } else {
        return a;
    }

    unreachable();
}

//
// inner_product
//

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto inner_product(IntTupleA const &a,
                                             IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        static_assert(tuple_size<IntTupleA>::value ==
                          tuple_size<IntTupleB>::value,
                      "Mismatched ranks");
        return transform_apply(
            a, b,
            [](auto const &x, auto const &y) { return inner_product(x, y); },
            [](auto const &...v) { return (Int<0>{} + ... + v); });
    } else {
        return a * b;
    }

    unreachable();
}

//
// ceil_div
//

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto ceil_div(IntTupleA const &a,
                                        IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value) {
        if constexpr (is_tuple<IntTupleB>::value) { // tuple tuple
            static_assert(tuple_size<IntTupleA>::value >=
                              tuple_size<IntTupleB>::value,
                          "Mismatched ranks");
            constexpr int R =
                tuple_size<IntTupleA>::value; // Missing ranks in TupleB are
                                              // implicitly 1
            return transform(
                a, append<R>(b, Int<1>{}),
                [](auto const &x, auto const &y) { return ceil_div(x, y); });
        } else { // tuple int
            auto const [result, rest] =
                fold(a, tal::make_tuple(tal::make_tuple(), b),
                     [](auto const &init, auto const &ai) {
                         return tal::make_tuple(
                             append(get<0>(init), ceil_div(ai, get<1>(init))),
                             ceil_div(get<1>(init), ai));
                     });
            return result;
        }
    } else if constexpr (is_tuple<IntTupleB>::value) { // int tuple
        return ceil_div(a, product(b));
    } else {
        return (a + b - Int<1>{}) / b;
    }

    unreachable();
}

//
// round_up
//   Round @a a up to the nearest multiple of @a b.
//   For negative numbers, rounds away from zero.
//

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto round_up(IntTupleA const &a,
                                        IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        static_assert(tuple_size<IntTupleA>::value >=
                          tuple_size<IntTupleB>::value,
                      "Mismatched ranks");
        constexpr int R =
            tuple_size<IntTupleA>::value; // Missing ranks in TupleB are
                                          // implicitly 1
        return transform(
            a, append<R>(b, Int<1>{}),
            [](auto const &x, auto const &y) { return round_up(x, y); });
    } else {
        return ((a + b - Int<1>{}) / b) * b;
    }

    unreachable();
}

/** Division for Shapes
 * Case Tuple Tuple:
 *   Perform shape_div element-wise
 * Case Tuple Int:
 *   Fold the division of b across each element of a
 *   Example: shape_div((4,5,6),40) -> shape_div((1,5,6),10) ->
 * shape_div((1,1,6),2) -> (1,1,3) Case Int Tuple: Return shape_div(a,
 * product(b)) Case Int Int: Enforce the divisibility condition a % b == 0 || b
 * % a == 0 when possible Return a / b with rounding away from 0 (that is, 1 or
 * -1 when a < b)
 */
template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto shape_div(IntTupleA const &a,
                                         IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value) {
        if constexpr (is_tuple<IntTupleB>::value) { // tuple tuple
            static_assert(tuple_size<IntTupleA>::value ==
                              tuple_size<IntTupleB>::value,
                          "Mismatched ranks");
            return transform(a, b, [](auto const &x, auto const &y) {
                return shape_div(x, y);
            });
        } else { // tuple int
            auto const [result, rest] =
                fold(a, tal::make_tuple(tal::make_tuple(), b),
                     [](auto const &init, auto const &ai) {
                         return tal::make_tuple(
                             append(get<0>(init), shape_div(ai, get<1>(init))),
                             shape_div(get<1>(init), ai));
                     });
            return result;
        }
    } else if constexpr (is_tuple<IntTupleB>::value) { // int tuple
        return shape_div(a, product(b));
    } else if constexpr (is_static<IntTupleA>::value &&
                         is_static<IntTupleB>::value) {
        static_assert(IntTupleA::value % IntTupleB::value == 0 ||
                          IntTupleB::value % IntTupleA::value == 0,
                      "Static shape_div failure");
        return C<shape_div(IntTupleA::value, IntTupleB::value)>{};
    } else { // int int
        // assert(a % b == 0 || b % a == 0);          // Waive dynamic assertion
        return a / b != 0
                   ? a / b
                   : signum(a) *
                         signum(b); // Division with rounding away from zero
    }

    unreachable();
}

/** Return a tuple the same profile as A scaled by corresponding elements in B
 */
template <class A, class B>
TAL_HOST_DEVICE constexpr auto elem_scale(A const &a, B const &b) {
    if constexpr (is_tuple<A>::value) {
        return transform(a, b, [](auto const &x, auto const &y) {
            return elem_scale(x, y);
        });
    } else {
        return a * product(b);
    }

    unreachable();
}

/** Replace the elements of Tuple B that are paired with an Int<0> with an
 * Int<1>
 */
template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto filter_zeros(IntTupleA const &a,
                                            IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value) {
        return transform(a, b, [](auto const &x, auto const &y) {
            return filter_zeros(x, y);
        });
    } else if constexpr (is_constant<0, IntTupleA>::value) {
        return repeat_like(b, Int<1>{});
    } else {
        return b;
    }

    unreachable();
}

template <class Tuple>
TAL_HOST_DEVICE constexpr auto filter_zeros(Tuple const &t) {
    return filter_zeros(t, t);
}

//
// Converters and constructors with arrays and params
//

/** Make an IntTuple of rank N from an Indexable array.
 * Access elements up to a dynamic index n, then use init (requires compatible
 * types) Consider tal::take<B,E> if all indexing is known to be valid \code
 *   std::vector<int> a = {6,3,4};
 *   auto tup = make_int_tuple<5>(a, a.size(), 0)            // (6,3,4,0,0)
 * \endcode
 */
template <int N, class Indexable, class T>
TAL_HOST_DEVICE constexpr auto make_int_tuple(Indexable const &t, int n,
                                              T const &init) {
    static_assert(N > 0);
    if constexpr (N == 1) {
        return 0 < n ? t[0] : init;
    } else {
        return transform(make_seq<N>{},
                         [&](auto i) { return i < n ? t[i] : init; });
    }

    unreachable();
}

/** Fill the dynamic values of a Tuple with values from another Tuple
 * \code
 *   auto params = make_tuple(6,3,4);
 *   tal::tuple<Int<1>, tal::tuple<int, int, Int<3>>, int, Int<2>> result;
 *   fill_int_tuple_from(result, params);                    //
 * (_1,(6,3,_3),4,_2) \endcode
 */
template <class Tuple, class TupleV>
TAL_HOST_DEVICE constexpr auto fill_int_tuple_from(Tuple &result,
                                                   TupleV const &vals) {
    return fold(result, vals, [](auto const &init, auto &&r) {
        if constexpr (is_static<remove_cvref_t<decltype(r)>>::
                          value) { // Skip static elements of result
            return init;
        } else if constexpr (is_tuple<remove_cvref_t<decltype(r)>>::
                                 value) { // Recurse into tuples
            return fill_int_tuple_from(r, init);
        } else { // Assign and consume arg
            static_assert(tuple_size<remove_cvref_t<decltype(init)>>::value > 0,
                          "Not enough values to fill with!");
            r = get<0>(init);
            return remove<0>(init);
        }

        unreachable();
    });
}

/** Make a "Tuple" by filling in the dynamic values in order from the arguments
 * \code
 *   using result_t = tal::tuple<Int<1>, tal::tuple<int, int, Int<3>>, int,
 * Int<2>>; auto result = make_int_tuple_from<result_t>(6,3,4);     //
 * (_1,(6,3,_3),4,_2) \endcode
 */
template <class Tuple, class... Ts>
TAL_HOST_DEVICE constexpr Tuple make_int_tuple_from(Ts const &...ts) {
    Tuple result = Tuple{};
    fill_int_tuple_from(result, tal::make_tuple(ts...));
    return result;
}

//
// Comparison operators
//

//
// There are many ways to compare tuple of elements and because CuTe is built
//   on parameterizing layouts of coordinates, some comparisons are appropriate
//   only in certain cases.
//  -- lexicographical comparison [reverse, reflected, revref]   : Correct for
//  coords in RowMajor Layout
//  -- colexicographical comparison [reverse, reflected, revref] : Correct for
//  coords in ColMajor Layout
//  -- element-wise comparison [any,all]                         :
// This can be very confusing. To avoid errors in selecting the appropriate
//   comparison, op<|op<=|op>|op>= are *not* implemented for tal::tuple.
//
// When actually desiring to order coordinates, the user should map them to
//   their indices within the Layout they came from:
//      e.g.  layoutX(coordA) < layoutX(coordB)
// That said, we implement the three most common ways to compare tuples below.
//   These are implemented with slighly more explicit names than op<.
//

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto lex_less(IntTupleA const &a, IntTupleB const &b);

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto colex_less(IntTupleA const &a,
                                          IntTupleB const &b);

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto elem_less(IntTupleA const &a,
                                         IntTupleB const &b);

namespace detail {

template <size_t I, class TupleA, class TupleB>
TAL_HOST_DEVICE constexpr auto lex_less_impl(TupleA const &a, TupleB const &b) {
    if constexpr (I == tuple_size<TupleB>::value) {
        return tal::false_type{}; // Terminal: TupleB is exhausted
    } else if constexpr (I == tuple_size<TupleA>::value) {
        return tal::true_type{}; // Terminal: TupleA is exhausted, TupleB is not
                                 // exhausted
    } else {
        return lex_less(get<I>(a), get<I>(b)) ||
               (get<I>(a) == get<I>(b) && lex_less_impl<I + 1>(a, b));
    }

    unreachable();
}

template <size_t I, class TupleA, class TupleB>
TAL_HOST_DEVICE constexpr auto colex_less_impl(TupleA const &a,
                                               TupleB const &b) {
    if constexpr (I == tuple_size<TupleB>::value) {
        return tal::false_type{}; // Terminal: TupleB is exhausted
    } else if constexpr (I == tuple_size<TupleA>::value) {
        return tal::true_type{}; // Terminal: TupleA is exhausted, TupleB is not
                                 // exhausted
    } else {
        constexpr size_t A = tuple_size<TupleA>::value - 1 - I;
        constexpr size_t B = tuple_size<TupleB>::value - 1 - I;
        return colex_less(get<A>(a), get<B>(b)) ||
               (get<A>(a) == get<B>(b) && colex_less_impl<I + 1>(a, b));
    }

    unreachable();
}

template <size_t I, class TupleA, class TupleB>
TAL_HOST_DEVICE constexpr auto elem_less_impl(TupleA const &a,
                                              TupleB const &b) {
    if constexpr (I == tuple_size<TupleA>::value) {
        return tal::true_type{}; // Terminal: TupleA is exhausted
    } else if constexpr (I == tuple_size<TupleB>::value) {
        return tal::false_type{}; // Terminal: TupleA is not exhausted, TupleB
                                  // is exhausted
    } else {
        return elem_less(get<I>(a), get<I>(b)) && elem_less_impl<I + 1>(a, b);
    }

    unreachable();
}

} // end namespace detail

// Lexicographical comparison

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto lex_less(IntTupleA const &a,
                                        IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        return detail::lex_less_impl<0>(a, b);
    } else {
        return a < b;
    }

    unreachable();
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto lex_leq(T const &t, U const &u) {
    return !lex_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto lex_gtr(T const &t, U const &u) {
    return lex_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto lex_geq(T const &t, U const &u) {
    return !lex_less(t, u);
}

// Colexicographical comparison

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto colex_less(IntTupleA const &a,
                                          IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        return detail::colex_less_impl<0>(a, b);
    } else {
        return a < b;
    }

    unreachable();
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto colex_leq(T const &t, U const &u) {
    return !colex_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto colex_gtr(T const &t, U const &u) {
    return colex_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto colex_geq(T const &t, U const &u) {
    return !colex_less(t, u);
}

// Elementwise [all] comparison

template <class IntTupleA, class IntTupleB>
TAL_HOST_DEVICE constexpr auto elem_less(IntTupleA const &a,
                                         IntTupleB const &b) {
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        return detail::elem_less_impl<0>(a, b);
    } else {
        return a < b;
    }

    unreachable();
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto elem_leq(T const &t, U const &u) {
    return !elem_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto elem_gtr(T const &t, U const &u) {
    return elem_less(u, t);
}

template <class T, class U>
TAL_HOST_DEVICE constexpr auto elem_geq(T const &t, U const &u) {
    return !elem_less(t, u);
}

} // namespace causalflow::tal