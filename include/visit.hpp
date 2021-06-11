#ifndef VISIT_HPP
#define VISIT_HPP

#include <cassert>
#include <type_traits>
#include <variant>

namespace detail {

template <typename> struct variant_trait {};

template <typename... Tys> struct variant_trait<std::variant<Tys...>> {
  using indexes = std::make_index_sequence<sizeof...(Tys)>;
};

template <typename ret_type, typename Func, typename Var>
[[noreturn]] inline ret_type
visit_single_impl(Func&&, std::integer_sequence<size_t>, Var&&) {
  assert(false && "unreachable");
  __builtin_unreachable();
}

template <typename ret_type, typename Func, typename Var, auto First,
          auto... Idx>
inline ret_type visit_single_impl(Func&& f,
                                  std::integer_sequence<size_t, First, Idx...>,
                                  Var&& var) {
  if (var.index() == First)
    return std::forward<Func>(f)(std::get<First>(var));
  return visit_single_impl<ret_type>(std::forward<Func>(f),
                                     std::integer_sequence<size_t, Idx...> {},
                                     std::forward<Var>(var));
}

template <typename Func, typename Var>
decltype(auto) visit_single(Func&& f, Var&& var) {
  assert((!var.valueless_by_exception()));
  using ret_type =
      std::invoke_result_t<Func, decltype(std::get<0>(std::declval<Var>()))>;
  return visit_single_impl<ret_type>(
      std::forward<Func>(f),
      typename variant_trait<
          std::remove_cv_t<std::remove_reference_t<Var>>>::indexes {},
      std::forward<Var>(var));
}
} // namespace detail

auto monostate_dispatch(auto&& dispatch, auto&& monostate_value) {
  return [&](auto&& arg) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(arg)>,
                                 std::monostate>) {
      assert(false && "Try to dispatch to monostate value");
      return monostate_value;
    } else {
      return dispatch(arg);
    }
  };
}

/// dev_visit is std::visit implementation suitable to be used in device code.
/// this version of visit doesn't use any function pointer but uses if series
/// which will be turned into switch case by the optimizer.
template <typename Func, typename Var, typename... Rest>
auto dev_visit(Func&& f, Var&& var, Rest&&... rest) {
  if constexpr (sizeof...(Rest) == 0)
    return detail::visit_single(std::forward<Func>(f), std::forward<Var>(var));
  else
    return detail::visit_single(
        [&](auto&& First) {
          return visit(
              [&](auto&&... Others) {
                std::forward<Func>(f)(
                    std::forward<decltype(First)>(First),
                    std::forward<decltype(Others)>(Others)...);
              },
              std::forward<Rest>(rest)...);
        },
        std::forward<Var>(var));
}

#endif // VISIT_HPP
