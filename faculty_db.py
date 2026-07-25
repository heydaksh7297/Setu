"""
JECRC Foundation - Dynamic Faculty Search System
Searches faculty by name, department, designation, specialization
With Fuzzy Name Matching for typo tolerance
"""

import json
import re
import os


class FacultyDB:
    """
    Dynamic Faculty Database with NLP-powered search
    - Search by name (fuzzy match)
    - Search by department
    - Search by designation (HOD, Professor etc.)
    - Search by specialization
    """

    def __init__(self, data_file='faculty_data.json'):
        """Initialize Faculty Database"""
        self.faculty_data = None
        self.all_faculty = []          # Flat list of all faculty
        self.all_names = []            # All faculty names (lowercase)
        self.name_to_faculty = {}      # name → faculty dict
        self.dept_aliases = {}         # alias → dept code

        self._load_data(data_file)
        self._build_index()

        print(f"👨‍🏫 FacultyDB initialized: {len(self.all_faculty)} faculty across {len(self.faculty_data.get('departments', {}))} departments")

    def _load_data(self, data_file):
        """Load faculty data from JSON"""
        try:
            filepath = os.path.join(os.path.dirname(__file__), data_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                self.faculty_data = json.load(f)
            print(f"✅ Faculty data loaded from {data_file}")
        except FileNotFoundError:
            print(f"❌ {data_file} not found!")
            self.faculty_data = {"departments": {}, "leadership": {}}
        except json.JSONDecodeError as e:
            print(f"❌ JSON error in {data_file}: {e}")
            self.faculty_data = {"departments": {}, "leadership": {}}

    def _build_index(self):
        """Build searchable index from faculty data"""
        departments = self.faculty_data.get('departments', {})

        # Department aliases for flexible search
        self.dept_aliases = {
            # CSE
            'cse': 'CSE',
            'computer science': 'CSE',
            'computer': 'CSE',
            'cs': 'CSE',
            'computer engineering': 'CSE',
            'computer science engineering': 'CSE',
            'computer science and engineering': 'CSE',
            'cse department': 'CSE',
            'computer department': 'CSE',
            'computer science department': 'CSE',

            # CSAI
            'csai': 'CSAI',
            'cs ai': 'CSAI',
            'cse ai': 'CSAI',
            'computer science ai': 'CSAI',
            'computer science artificial intelligence': 'CSAI',
            'cse artificial intelligence': 'CSAI',
            'artificial intelligence cse': 'CSAI',
            'ai branch': 'CSAI',
            'csai department': 'CSAI',

            # AIDS
            'aids': 'AIDS',
            'ai ds': 'AIDS',
            'ai and ds': 'AIDS',
            'artificial intelligence and data science': 'AIDS',
            'artificial intelligence data science': 'AIDS',
            'ai data science': 'AIDS',
            'data science': 'AIDS',
            'aids department': 'AIDS',

            # IT
            'it': 'IT',
            'information technology': 'IT',
            'info tech': 'IT',
            'it department': 'IT',
            'information technology department': 'IT',

            # ECE
            'ece': 'ECE',
            'electronics': 'ECE',
            'electronics and communication': 'ECE',
            'electronics communication': 'ECE',
            'ec': 'ECE',
            'electronics engineering': 'ECE',
            'ece department': 'ECE',
            'electronics department': 'ECE',

            # EE
            'ee': 'EE',
            'electrical': 'EE',
            'electrical engineering': 'EE',
            'electrical department': 'EE',
            'ee department': 'EE',

            # ME
            'me': 'ME',
            'mechanical': 'ME',
            'mechanical engineering': 'ME',
            'mech': 'ME',
            'mechanical department': 'ME',
            'me department': 'ME',

            # CE
            'ce': 'CE',
            'civil': 'CE',
            'civil engineering': 'CE',
            'civil department': 'CE',
            'ce department': 'CE',

            # FIRST YEAR
            'first year': 'FIRST_YEAR',
            'first_year': 'FIRST_YEAR',
            'firstyear': 'FIRST_YEAR',
            '1st year': 'FIRST_YEAR',
            '1styear': 'FIRST_YEAR',
            'fy': 'FIRST_YEAR',
            'first': 'FIRST_YEAR',
            'pehla saal': 'FIRST_YEAR',
            'pehla year': 'FIRST_YEAR',
            'pratham varsh': 'FIRST_YEAR',
            'first year department': 'FIRST_YEAR',
            'common faculty': 'FIRST_YEAR',
            'first year faculty': 'FIRST_YEAR',
            'first year common': 'FIRST_YEAR',
        }
        # Build flat list of all faculty
        self.all_faculty = []
        self.all_names = []
        self.name_to_faculty = {}

        for dept_code, dept_data in departments.items():
            faculty_list = dept_data.get('faculty', [])
            for faculty in faculty_list:
                faculty_entry = {
                    **faculty,
                    'department': dept_code,
                    'department_full': dept_data.get('full_name', dept_code),
                    'is_hod': faculty.get('name', '') == dept_data.get('hod', '')
                }
                self.all_faculty.append(faculty_entry)

                # Index by name variations
                name = faculty.get('name', '').lower()
                self.all_names.append(name)
                self.name_to_faculty[name] = faculty_entry

                # Also index without prefix (Dr., Prof., etc.)
                clean_name = re.sub(r'^(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?)\s*', '', name).strip()
                self.name_to_faculty[clean_name] = faculty_entry

                # Index by last name only
                parts = clean_name.split()
                if len(parts) > 1:
                    self.name_to_faculty[parts[-1]] = faculty_entry
                    # First name too
                    self.name_to_faculty[parts[0]] = faculty_entry

        # Index leadership
        leadership = self.faculty_data.get('leadership', {})
        for role, person in leadership.items():
            name = person.get('name', '').lower()
            self.name_to_faculty[name] = {**person, 'role': role, 'is_leadership': True}
            clean_name = re.sub(r'^(shri|dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?)\s*', '', name).strip()
            self.name_to_faculty[clean_name] = {**person, 'role': role, 'is_leadership': True}
            parts = clean_name.split()
            if parts:
                self.name_to_faculty[parts[-1]] = {**person, 'role': role, 'is_leadership': True}

    # ═══════════════════════════════════════
    # 🔍 Fuzzy Name Matching
    # ═══════════════════════════════════════
    def _levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _fuzzy_find_name(self, query_name, max_distance=2):
        query_lower = query_name.lower().strip()
        query_clean = re.sub(
            r'^(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|shri)\s*',
            '', query_lower
        ).strip()

        # Skip if query is a role word, not a name
        role_words = {'principal', 'chairman', 'director', 'dean',
                      'hod', 'head', 'vice', 'professor', 'teacher',
                      'faculty', 'sir', 'madam', 'mam'}
        if query_clean in role_words:
            return None

        # Exact match
        for check in [query_lower, query_clean]:
            if check in self.name_to_faculty:
                return self.name_to_faculty[check]

        # Partial match — STRICT: both must be 4+ chars and overlap > 70%
        for indexed_name, faculty in self.name_to_faculty.items():
            if len(query_clean) >= 4 and len(indexed_name) >= 4:
                if query_clean in indexed_name or indexed_name in query_clean:
                    min_len = min(len(query_clean), len(indexed_name))
                    max_len = max(len(query_clean), len(indexed_name))
                    overlap = min_len / max_len
                    if overlap >= 0.7:
                        return faculty

        # Fuzzy match — only for 4+ char names
        best_match = None
        best_distance = float('inf')

        for indexed_name, faculty in self.name_to_faculty.items():
            if len(indexed_name) < 4 or len(query_clean) < 4:
                continue
            if abs(len(indexed_name) - len(query_clean)) > max_distance:
                continue
            dist = self._levenshtein_distance(query_clean, indexed_name)
            if dist < best_distance and dist <= max_distance:
                best_distance = dist
                best_match = faculty

        return best_match
    # ═══════════════════════════════════════
    # 🔍 Search Methods
    # ═══════════════════════════════════════
    def search_by_name(self, name):
        """Search faculty by name (with fuzzy matching)"""
        result = self._fuzzy_find_name(name)
        if result:
            return self._format_faculty_response(result)
        return None

    def search_by_specialization(self, spec_query):
        """Search faculty by specialization (maths, physics, chemistry etc.)"""
        spec_query_lower = spec_query.lower().strip()

        # Specialization aliases
        spec_aliases = {
            'maths': 'mathematics', 'math': 'mathematics',
            'mathematics': 'mathematics', 'ganit': 'mathematics',
            'physics': 'physics', 'bhautiki': 'physics',
            'chemistry': 'chemistry', 'chem': 'chemistry',
            'rasayan': 'chemistry',
            'english': 'english & humanities', 'humanities': 'english & humanities',
            'english & humanities': 'english & humanities',
            'angrezi': 'english & humanities',
            'mechanical': 'mechanical engineering', 'mech': 'mechanical engineering',
            'mechanical engineering': 'mechanical engineering',
            'electrical': 'electrical engineering',
            'electrical engineering': 'electrical engineering',
            'civil': 'civil engineering',
            'civil engineering': 'civil engineering',
            'computer': 'computer science and engineering',
            'computer science': 'computer science and engineering',
            'computer science and engineering': 'computer science and engineering',
            'sports': 'sports', 'khel': 'sports',
        }

        resolved_spec = spec_aliases.get(spec_query_lower, spec_query_lower)

        # Search all faculty — ONLY match if specialization field EXISTS and MATCHES
        matching_faculty = []
        for faculty in self.all_faculty:
            faculty_spec = faculty.get('specialization', '').lower().strip()

            # Skip if specialization is empty
            if not faculty_spec:
                continue

            # Match only if specialization actually matches
            if resolved_spec == faculty_spec or resolved_spec in faculty_spec:
                matching_faculty.append(faculty)

        if not matching_faculty:
            return None

        return self._format_specialization_response(spec_query, resolved_spec, matching_faculty)

    def _format_department_response(self, dept_code, dept_data):
        """Format department faculty list with full details"""
        full_name = dept_data.get('full_name', dept_code)
        hod = dept_data.get('hod', 'N/A')
        faculty_list = dept_data.get('faculty', [])

        lines = [
            f"🏛️ **{full_name} ({dept_code}) — Faculty List:**",
            f"",
            f"👑 **HOD/Head:** {hod}",
            f"👨‍🏫 **Total Faculty:** {len(faculty_list)}",
            f"",
        ]

        groups = {}
        for faculty in faculty_list:
            desig = faculty.get('designation', 'Other')
            if desig not in groups:
                groups[desig] = []
            groups[desig].append(faculty)

        desig_order = [
            'Dean - 1st Year', 'Principal',
            'Professor', 'Associate Professor',
            'Assistant Professor'
        ]

        shown_desigs = set()
        for desig in desig_order:
            if desig in groups:
                shown_desigs.add(desig)
                emoji = '👑' if 'Dean' in desig or 'Principal' in desig else '🔹'
                lines.append(f"**{emoji} {desig}s ({len(groups[desig])}):**")
                for f in groups[desig]:
                    name = f.get('name', '')
                    qual = f.get('qualification', '')
                    spec = f.get('specialization', '')
                    doj = f.get('date_of_joining', '')
                    is_hod = name == hod
                    hod_tag = " 👑" if is_hod else ""
                    spec_tag = f" | 📚 {spec}" if spec else ""
                    doj_tag = f" | 📅 {doj}" if doj else ""
                    lines.append(f"  • **{name}**{hod_tag} — {qual}{spec_tag}{doj_tag}")
                lines.append("")

        for desig, members in groups.items():
            if desig not in shown_desigs:
                lines.append(f"**Others ({len(members)}):**")
                for f in members:
                    name = f.get('name', '')
                    qual = f.get('qualification', '')
                    spec = f.get('specialization', '')
                    doj = f.get('date_of_joining', '')
                    is_hod = name == hod
                    hod_tag = " 👑" if is_hod else ""
                    spec_tag = f" | 📚 {spec}" if spec else ""
                    doj_tag = f" | 📅 {doj}" if doj else ""
                    lines.append(f"  • **{name}**{hod_tag} — {qual}{spec_tag}{doj_tag}")
                lines.append("")

        lines.append(f"💡 Know more about anyone:")
        lines.append(f"  _\"Tell me about {faculty_list[0].get('name', '')}\"_")

        return '\n'.join(lines)

    def _format_specialization_response(self, query, spec_name, faculty_list):
        """Format faculty list filtered by specialization"""
        lines = [
            f"📚 **Faculty — Specialization: {spec_name.title()}**",
            f"",
            f"👨‍🏫 **Total Faculty Found:** {len(faculty_list)}",
            f"",
        ]

        # Group by designation
        groups = {}
        for f in faculty_list:
            desig = f.get('designation', 'Other')
            if desig not in groups:
                groups[desig] = []
            groups[desig].append(f)

        desig_order = [
            'Dean - 1st Year', 'Principal',
            'Professor', 'Associate Professor',
            'Assistant Professor'
        ]

        shown = set()
        for desig in desig_order:
            if desig in groups:
                shown.add(desig)
                lines.append(f"**🔹 {desig}s ({len(groups[desig])}):**")
                for f in groups[desig]:
                    name = f.get('name', '')
                    qual = f.get('qualification', '')
                    dept = f.get('department', '')
                    doj = f.get('date_of_joining', '')
                    gender = f.get('gender', '')
                    gender_emoji = "👩" if gender == "Female" else "👨"
                    lines.append(f"  {gender_emoji} **{name}** — {qual} | 🏛️ {dept} | 📅 {doj}")
                lines.append("")

        for desig, members in groups.items():
            if desig not in shown:
                lines.append(f"**🔹 {desig} ({len(members)}):**")
                for f in members:
                    name = f.get('name', '')
                    qual = f.get('qualification', '')
                    dept = f.get('department', '')
                    doj = f.get('date_of_joining', '')
                    gender = f.get('gender', '')
                    gender_emoji = "👩" if gender == "Female" else "👨"
                    lines.append(f"  {gender_emoji} **{name}** — {qual} | 🏛️ {dept} | 📅 {doj}")
                lines.append("")

        lines.append(f"💡 Know more: _\"Tell me about [name]\"_")

        return '\n'.join(lines)

    def search_by_department(self, dept_query):
        """Get all faculty of a department"""
        dept_query_lower = dept_query.lower().strip()

        # ⭐ Block subject words from being treated as departments
        subject_words = {
            'physics', 'chemistry', 'maths', 'math', 'mathematics',
            'english', 'humanities', 'sports', 'ganit', 'bhautiki',
            'rasayan', 'angrezi', 'khel', 'chem',
        }
        if dept_query_lower in subject_words:
            print(f"    ⚠️ '{dept_query}' is a subject, not department. Skipping dept search.")
            return None

        # ... rest of existing code ...
        dept_query_lower = dept_query.lower().strip()

        # Remove common suffixes
        dept_query_clean = re.sub(r'\s*(department|dept|branch|faculty|faculties|teachers?|staff|list|details|info)\s*', '', dept_query_lower).strip()

        # Try original query
        dept_code = self.dept_aliases.get(dept_query_lower)

        # Try cleaned query
        if not dept_code:
            dept_code = self.dept_aliases.get(dept_query_clean)

        # Try uppercase
        if not dept_code:
            dept_code = dept_query_clean.upper()
            if dept_code not in self.faculty_data.get('departments', {}):
                dept_code = None

        # Try partial match in aliases
        if not dept_code:
            for alias, code in self.dept_aliases.items():
                if dept_query_clean in alias or alias in dept_query_clean:
                    if len(dept_query_clean) >= 2 and len(alias) >= 2:
                        dept_code = code
                        break

        if not dept_code:
            print(f"    ❌ Department not found: '{dept_query}' (cleaned: '{dept_query_clean}')")
            return None

        departments = self.faculty_data.get('departments', {})
        if dept_code in departments:
            dept_data = departments[dept_code]
            print(f"    ✅ Department found: {dept_code} ({len(dept_data.get('faculty', []))} faculty)")
            return self._format_department_response(dept_code, dept_data)

        return None

    def search_hod(self, dept_query=None):
        """Get HOD of a department (or all HODs)"""
        departments = self.faculty_data.get('departments', {})

        if dept_query:
            dept_query_lower = dept_query.lower().strip()

            # Clean department query
            dept_query_clean = re.sub(r'\s*(department|dept|branch|faculty|faculties)\s*', '', dept_query_lower).strip()

            # Try alias
            dept_code = self.dept_aliases.get(dept_query_lower)
            if not dept_code:
                dept_code = self.dept_aliases.get(dept_query_clean)
            if not dept_code:
                upper = dept_query_clean.upper()
                if upper in departments:
                    dept_code = upper

            # Partial match
            if not dept_code:
                for alias, code in self.dept_aliases.items():
                    if dept_query_clean in alias or alias in dept_query_clean:
                        if len(dept_query_clean) >= 2:
                            dept_code = code
                            break

            if dept_code and dept_code in departments:
                dept_data = departments[dept_code]
                hod_name = dept_data.get('hod', '')
                print(f"    👑 HOD found: {hod_name} ({dept_code})")

                for faculty in dept_data.get('faculty', []):
                    if faculty.get('name') == hod_name:
                        faculty_entry = {
                            **faculty,
                            'department': dept_code,
                            'department_full': dept_data.get('full_name', ''),
                            'is_hod': True
                        }
                        return self._format_faculty_response(faculty_entry)

                # HOD name found but not in faculty list
                return f"👑 **HOD of {dept_data.get('full_name', dept_code)}:** {hod_name}"

            print(f"    ❌ HOD dept not found: '{dept_query}'")
            return None
        else:
            return self._format_all_hods(departments)

    def search_leadership(self, role=None):
        """Get leadership info (chairman, director, principal, dean)"""
        leadership = self.faculty_data.get('leadership', {})

        if role:
            role_lower = role.lower().strip()
            role_aliases = {
                'chairman': 'chairman', 'chairperson': 'chairman',
                'vice chairman': 'vice_chairperson', 'vice chairperson': 'vice_chairperson',
                'vc': 'vice_chairperson',
                'director': 'director',
                'principal': 'principal',
                'dean': 'dean_first_year', 'dean first year': 'dean_first_year',
                'dean 1st year': 'dean_first_year',
            }
            resolved_role = role_aliases.get(role_lower, role_lower)
            if resolved_role in leadership:
                return self._format_leadership_response(resolved_role, leadership[resolved_role])
        else:
            return self._format_all_leadership(leadership)

        return None

    def get_department_list(self):
        """Get list of all departments"""
        departments = self.faculty_data.get('departments', {})
        lines = ["🏛️ **Departments at JECRC Foundation:**\n"]
        for dept_code, dept_data in departments.items():
            full_name = dept_data.get('full_name', dept_code)
            hod = dept_data.get('hod', 'N/A')
            count = len(dept_data.get('faculty', []))
            lines.append(f"🔹 **{dept_code}** — {full_name}")
            lines.append(f"   👨‍🏫 HOD: {hod} | Faculty: {count}")
        lines.append(f"\n💡 Ask: _\"CSE faculty list\"_ or _\"Who is HOD of IT?\"_")
        return '\n'.join(lines)

    # ═══════════════════════════════════════
    # 📝 Response Formatters
    # ═══════════════════════════════════════
    def _format_faculty_response(self, faculty):
        """Format a single faculty member's info with ALL fields"""
        if faculty.get('is_leadership'):
            return self._format_leadership_response(
                faculty.get('role', ''),
                faculty
            )

        name = faculty.get('name', 'N/A')
        designation = faculty.get('designation', 'N/A')
        dept = faculty.get('department', '')
        dept_full = faculty.get('department_full', dept)
        qualification = faculty.get('qualification', 'N/A')
        specialization = faculty.get('specialization', '')
        gender = faculty.get('gender', '')
        doj = faculty.get('date_of_joining', '')
        email = faculty.get('email', '')
        experience = faculty.get('experience', '')
        publications = faculty.get('publications', '')
        achievements = faculty.get('achievements', '')
        is_hod = faculty.get('is_hod', False)

        hod_badge = " 🏆 (HOD)" if is_hod else ""
        gender_emoji = "👩‍🏫" if gender == "Female" else "👨‍🏫"

        lines = [
            f"{gender_emoji} **{name}**{hod_badge}",
            f"",
            f"🔹 **Designation:** {designation}",
            f"🔹 **Department:** {dept_full} ({dept})",
            f"🔹 **Qualification:** {qualification}",
        ]

        if specialization:
            lines.append(f"🔹 **Specialization:** {specialization}")
        if gender:
            lines.append(f"🔹 **Gender:** {gender}")
        if doj:
            lines.append(f"🔹 **Date of Joining:** {doj}")
        if experience:
            lines.append(f"🔹 **Experience:** {experience}")
        if publications:
            lines.append(f"🔹 **Publications:** {publications}")
        if email and email != 'N/A':
            lines.append(f"🔹 **Email:** {email}")
        if achievements:
            lines.append(f"🔹 **Achievements:** {achievements}")

        lines.append(f"\n💡 Ask more: _\"{dept} faculty list\"_ or _\"{dept} HOD\"_")

        return '\n'.join(lines)
    def _format_all_hods(self, departments):
        """Format list of all HODs"""
        lines = ["👑 **HODs at JECRC Foundation:**\n"]
        for dept_code, dept_data in departments.items():
            full_name = dept_data.get('full_name', dept_code)
            hod = dept_data.get('hod', 'N/A')
            lines.append(f"🔹 **{dept_code}** ({full_name}): **{hod}**")
        lines.append(f"\n💡 Ask: _\"Tell me about Dr. XYZ\"_ for detailed info")
        return '\n'.join(lines)

    def _format_leadership_response(self, role, person):
        """Format leadership info"""
        name = person.get('name', 'N/A')
        designation = person.get('designation', 'N/A')
        qualification = person.get('qualification', '')
        about = person.get('about', '')
        experience = person.get('experience', '')

        role_emoji = {
            'chairman': '👔',
            'vice_chairman': '👔',
            'director': '🎓',
            'principal': '🎓'
        }
        emoji = role_emoji.get(role, '👤')

        lines = [
            f"{emoji} **{name}**",
            f"",
            f"🔹 **Designation:** {designation}",
        ]

        if qualification:
            lines.append(f"🔹 **Qualification:** {qualification}")
        if about:
            lines.append(f"🔹 **About:** {about}")
        if experience:
            lines.append(f"🔹 **Experience:** {experience}")

        return '\n'.join(lines)

    def _format_all_leadership(self, leadership):
        """Format all leadership"""
        lines = ["👔 **JECRC Foundation Leadership:**\n"]
        for role, person in leadership.items():
            name = person.get('name', 'N/A')
            designation = person.get('designation', 'N/A')
            lines.append(f"🔹 **{designation}:** {name}")
        lines.append(f"\n💡 Ask: _\"Tell me about chairman\"_ for detailed info")
        return '\n'.join(lines)

    # ═══════════════════════════════════════
    # 🧠 Smart Query Parser
    # ═══════════════════════════════════════
    def parse_and_search(self, user_message):
        """
        Smart parser - checks in correct priority order:
        1. Leadership/Dean (principal, chairman, dean etc.)
        2. All HODs
        3. Specific HOD search
        4. Specialization search (maths, physics etc.)
        5. Department faculty list
        6. Name with prefix (Dr., Prof.)
        7. "Tell me about [Name]"
        8. Direct name detection
        """
        msg_lower = user_message.lower().strip()

        # Clean common prefixes
        msg_cleaned = re.sub(
            r'^(tell\s+me\s+about|who\s+is|about|info\s+about|details\s+of|details\s+about)\s+',
            '', msg_lower
        ).strip()

        # ══════════════════════════════════════
        # PRIORITY 1: Leadership + Dean search
        # ══════════════════════════════════════
        leadership_keywords = {
            'principal': 'principal',
            'chairman': 'chairman', 'chairperson': 'chairman',
            'vice chairman': 'vice_chairperson',
            'vice chairperson': 'vice_chairperson',
            'director': 'director',
            'dean of first year': 'dean_first_year',
            'first year dean': 'dean_first_year',
            'dean first year': 'dean_first_year',
            'dean 1st year': 'dean_first_year',
            'dean mam': 'dean_first_year',
            'dean maam': 'dean_first_year',
            'dean madam': 'dean_first_year',
            'dean sir': 'dean_first_year',
            'fy dean': 'dean_first_year',
            'first year ki dean': 'dean_first_year',
            'first year ka dean': 'dean_first_year',
            'first year ke dean': 'dean_first_year',
            'dean': 'dean_first_year',
        }

        for keyword, role in leadership_keywords.items():
            if keyword in msg_lower:
                result = self.search_leadership(role)
                if result:
                    return result, 'leadership_search'

        # ══════════════════════════════════════
        # PRIORITY 2: All HODs
        # ══════════════════════════════════════
        if any(w in msg_lower for w in ['all hod', 'sab hod', 'sabhi hod', 'all hods', 'sab hods']):
            result = self.search_hod()
            if result:
                return result, 'all_hods'

        if msg_lower.strip() in ['hod', 'hods']:
            result = self.search_hod()
            if result:
                return result, 'all_hods'

        # ══════════════════════════════════════
        # PRIORITY 3: Specific HOD search
        # ══════════════════════════════════════
        hod_patterns = [
            r'(?:hod|head\s+of\s+department|head)\s+(?:of\s+)?(\w+(?:\s+\w+)?)\s*(?:branch|department|dept)?',
            r'(\w+(?:\s+\w+)?)\s+(?:branch|department|dept)\s*(?:ka|ki|ke)?\s*(?:hod|head)',
            r'(\w+(?:\s+\w+)?)\s+(?:ka|ki|ke)\s+hod',
            r'(\w+)\s+hod',
            r'(?:who\s+is\s+)?hod\s+(?:of\s+)?(\w+(?:\s+\w+)?)',
        ]

        for pattern in hod_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                dept_query = None
                for g in match.groups():
                    if g:
                        dept_query = g.strip()
                if dept_query:
                    dept_query = re.sub(r'\b(of|the|is|ka|ki|ke|branch|department|dept|who)\b', '', dept_query).strip()
                    if dept_query and len(dept_query) >= 2:
                        print(f"    👑 HOD query: '{dept_query}'")
                        result = self.search_hod(dept_query)
                        if result:
                            return result, 'hod_search'

        # ══════════════════════════════════════
        # PRIORITY 4: Specialization search
        # (maths teacher, physics faculty etc.)
        # MUST BE BEFORE department search!
        # ══════════════════════════════════════
        spec_keywords = [
            'maths', 'math', 'mathematics', 'ganit',
            'physics', 'bhautiki',
            'chemistry', 'chem', 'rasayan',
            'english', 'humanities', 'angrezi',
            'sports', 'khel',
        ]

        teacher_words = [
            'teacher', 'teachers', 'faculty', 'faculties',
            'professor', 'professors', 'staff', 'sir', 'madam',
            'wale', 'wali', 'padhane', 'sikhane',
        ]

        has_spec = None
        for kw in spec_keywords:
            if kw in msg_lower:
                has_spec = kw
                break

        has_teacher = any(tw in msg_lower for tw in teacher_words)

        if has_spec:
            # Method 1: Direct keyword + teacher word combo
            if has_teacher:
                print(f"    📚 Specialization direct: '{has_spec}'")
                result = self.search_by_specialization(has_spec)
                if result:
                    return result, 'specialization_search'

            # Method 2: Regex patterns
            spec_patterns = [
                r'(maths?|mathematics|physics|chemistry|chem|english|humanities|sports|ganit|bhautiki|rasayan)\s+(?:teacher|teachers|faculty|faculties|professor|professors|staff|sir|madam|wale|ke\s+teacher)',
                r'(?:teacher|teachers|faculty|faculties|professor|professors|staff)\s+(?:of|for|in)\s+(maths?|mathematics|physics|chemistry|chem|english|humanities|sports)',
                r'(maths?|mathematics|physics|chemistry|chem|english|humanities|sports)\s+(?:ke|ki|ka)\s+(?:teacher|faculty|sir|madam)',
                r'(?:who\s+teaches?|kaun\s+padhata|kaun\s+padhati|kaun\s+padhaate)\s+(maths?|mathematics|physics|chemistry|english)',
                r'(maths?|mathematics|physics|chemistry|chem|english|humanities|sports)\s+(?:padhane|sikhane|wale)',
            ]

            for pattern in spec_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    spec_query = match.group(1).strip()
                    print(f"    📚 Specialization regex: '{spec_query}'")
                    result = self.search_by_specialization(spec_query)
                    if result:
                        return result, 'specialization_search'

        # ══════════════════════════════════════
        # PRIORITY 4.5: Direct department + faculty keyword
        # Handles: "ME faculty list", "CE teachers", "EE staff"
        # ══════════════════════════════════════
        direct_dept_keywords = {
            'cse', 'csai', 'aids', 'it', 'ee', 'ece', 'ce', 'me',
            'first year', '1st year', 'fy',
            'mechanical', 'electrical', 'civil', 'electronics',
            'computer science', 'information technology',
            'computer', 'artificial intelligence',
        }

        faculty_words = [
            'faculty', 'faculties', 'teacher', 'teachers',
            'professor', 'professors', 'staff', 'list',
            'faculty list', 'teacher list',
        ]

        for dept_kw in direct_dept_keywords:
            if dept_kw in msg_lower:
                has_faculty_word = any(fw in msg_lower for fw in faculty_words)
                if has_faculty_word:
                    print(f"    📂 Direct dept keyword: '{dept_kw}'")
                    result = self.search_by_department(dept_kw)
                    if result:
                        return result, 'department_faculty'

        # ══════════════════════════════════════
        # PRIORITY 5: Department faculty list
        # ══════════════════════════════════════
        dept_patterns = [
            r'(?:faculties|faculty|teachers?|professors?|staff)\s+(?:of|in|from|for)\s+(.+?)(?:\s+department|\s+dept|\s+branch)?$',
            r'(.+?)\s+(?:faculties|faculty|teachers?|professors?|staff)\s*(?:list|details|info)?$',
            r'(.+?)\s+(?:department|dept|branch)\s+(?:faculties|faculty|teachers?|staff)',
            r'(?:list|all|show|display|get)\s+(?:faculties|faculty|teachers?)\s+(?:of|in|from)\s+(.+)',
            r'(.+?)\s+(?:department|dept)\s+(?:list|details|info)',
            r'(.+?)\s+ke\s+(?:faculties|faculty|teachers?|professors?)',
            r'(.+?)\s+ki\s+(?:faculties|faculty|teachers?)',
            r'(.+?)\s+ka\s+(?:faculties|faculty|teachers?|staff)',
        ]

        for pattern in dept_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                dept_query = match.group(1).strip()
                dept_query = re.sub(r'\b(the|of|in|from|all|show|list|detail|details|about|me|tell|give|get)\b', '', dept_query).strip()
                if dept_query and len(dept_query) >= 2:
                    print(f"    📂 Dept query extracted: '{dept_query}'")
                    result = self.search_by_department(dept_query)
                    if result:
                        return result, 'department_faculty'

        # ══════════════════════════════════════
        # PRIORITY 6: Name with prefix (Dr., Prof.)
        # ══════════════════════════════════════
        name_prefix_match = re.search(
            r'(dr\.?|prof\.?|professor|shri)\s+(\w[\w\s]*)',
            msg_lower
        )
        if name_prefix_match:
            full_match = name_prefix_match.group(0).strip()
            full_match = re.sub(
                r'\s+(sir|madam|ji|sahab|mam|faculty|teacher|professor|ke\s+baare|about).*$',
                '', full_match
            ).strip()
            if len(full_match) >= 4:
                result = self.search_by_name(full_match)
                if result:
                    return result, 'name_search'

        # ══════════════════════════════════════
        # PRIORITY 7: "Tell me about [Name]"
        # ══════════════════════════════════════
        about_patterns = [
            r'(?:tell\s+me\s+about|who\s+is|about|info\s+about|details\s+of)\s+(.+)',
            r'(.+?)\s+(?:ke\s+baare\s+mein|ke\s+bare\s+me|ke\s+baare|about)\s*(?:batao|bataiye)?',
            r'(.+?)\s+(?:kaun\s+hai|kon\s+hai|kya\s+hai)',
        ]

        for pattern in about_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                potential_name = match.group(1).strip()
                potential_name = re.sub(
                    r'\s+(sir|madam|ji|sahab|mam|faculty|teacher)$',
                    '', potential_name
                ).strip()

                role_words = {'principal', 'chairman', 'director', 'dean',
                              'hod', 'head', 'vice'}
                if potential_name in role_words:
                    continue

                if len(potential_name) >= 3:
                    result = self.search_by_name(potential_name)
                    if result:
                        return result, 'name_search'

        # ══════════════════════════════════════
        # PRIORITY 8: Department list
        # ══════════════════════════════════════
        if any(w in msg_lower for w in ['department list', 'all department', 'all departments', 'sab department', 'sabhi department']):
            return self.get_department_list(), 'department_list'

        # ══════════════════════════════════════
        # PRIORITY 9: Direct name detection
        # ══════════════════════════════════════
        words = msg_lower.split()
        for window_size in [3, 2]:
            for i in range(len(words) - window_size + 1):
                potential_name = ' '.join(words[i:i + window_size]).strip()

                skip_words = {'tell', 'me', 'about', 'who', 'is', 'the',
                              'sir', 'madam', 'ji', 'ke', 'ka', 'ki',
                              'hai', 'hod', 'faculty', 'teacher', 'department',
                              'batao', 'bataiye', 'kya', 'kaun', 'kon',
                              'principal', 'chairman', 'director', 'dean',
                              'head', 'vice'}

                if potential_name in skip_words:
                    continue

                if len(potential_name) >= 4:
                    if potential_name in self.name_to_faculty:
                        faculty = self.name_to_faculty[potential_name]
                        result = self._format_faculty_response(faculty)
                        return result, 'name_detected'

        return None, None


# ── Testing ──
if __name__ == "__main__":
    print("=" * 60)
    print("  Faculty DB — Test")
    print("=" * 60)

    db = FacultyDB()

    test_queries = [
        "Tell me about Dr. Amit Sharma",
        "CSE faculty list",
        "Who is HOD of IT?",
        "all HODs",
        "chairman kaun hai",
        "principal",
        "Dr. Priya Gupta",
        "Sneha Verma",
        "ECE teachers",
        "ME department faculty",
        "physics teachers",
        "maths faculties",
        "chemistry faculty",
        "hod of ee",
        "hod of ee branch",
        "first year faculties",
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        result, search_type = db.parse_and_search(query)
        if result:
            print(f"  ✅ Type: {search_type}")
            print(f"  Response:\n{result[:200]}...")
        else:
            print(f"  ❌ No result")
